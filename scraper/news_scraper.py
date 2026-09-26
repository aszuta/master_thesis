from bs4 import BeautifulSoup, SoupStrainer
import requests
import random
import datetime
import json
from utils.db_utils import init_db, get_conn, insert_news
import time

URL = 'https://www.bankier.pl/gielda/notowania/akcje/{}/wiadomosci'
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/115.0"
]

_TICKERS = ['PKN', 'CDR', 'PKO', 'KGH', 'LPP']
_SLUGS = {'PKN': 'pknorlen', 'CDR': 'cdprojekt', 'PKO': 'pkobp', 'KGH': 'kghm', 'LPP': 'lpp'}

_DATE_FMT = '%Y-%m-%d %H:%M'
DATE_START = datetime.date(2025, 12, 31)
DATE_END = datetime.date(2025, 12, 9)

ONLY_LIST = SoupStrainer("section", id="listing-article-list-box")

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

# pobieranie wiadomości z listy
def fetch_news(ticker):
    base_url = URL.format(_SLUGS[ticker])
    news = []
    page_num = 1

    while True:
        page = base_url if page_num == 1 else f'{base_url}/{page_num}'

        resp = requests.get(page, headers=get_headers())
        soup = BeautifulSoup(resp.text, 'lxml', parse_only=ONLY_LIST)
        rows = soup.find_all('li', class_='m-listing-article-list__item')
        if not rows:
            break

        for row in rows:
            link = row.find('a', class_='m-listing-article-list__anchor', href=True)
            date_div = row.find('div', class_='m-listing-article-list__date-time')
            if date_div is None:
                continue

            date = date_div.get_text(strip=True)

            title = row.find('div', class_='m-listing-article-list__title').get_text(strip=True)
            
            try:
                news_datetime = datetime.datetime.strptime(date, _DATE_FMT)
                news_date = news_datetime.date()

                if news_date > DATE_START:
                    continue

                if news_date < DATE_END:
                    return news
            except ValueError:
                continue

            news_url = link.get('href')
            if not news_url.startswith("http"):
                news_url = "https://www.bankier.pl" + news_url

            news_body = fetch_news_content(news_url)

            news.append({
                'url': news_url,
                'ticker': ticker,
                'timestamp': news_datetime.isoformat(sep=' '),
                'title': title,
                'lead': news_body['lead'],
                'full_text': news_body['lead'] + " " + " ".join(news_body['body'])
            })
        
        page_num += 1

    return news

# pobieranie zawartości wiadomości
def fetch_news_content(url, retries = 10, delay = 5):
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=get_headers())
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[ERROR] {url} (try {attempt}/{retries}): {e}")
            if attempt == retries:
                return {"lead": "", "body": []}
            time.sleep(delay)
            continue

        soup = BeautifulSoup(resp.text, 'lxml')
        article = soup.find('section', class_='o-article-content')
        
        if article is None:
            time.sleep(delay)
            continue

        lead_tag = article.find('span', class_='lead')
        lead = lead_tag.get_text(strip=True) if lead_tag else ""
            
        body = []
        for p in article.find_all('p'):
            text = p.get_text(strip=True)
            if not text:
                continue

            if lead and text.startswith(lead):
                continue

            body.append(text)

        return {
            "lead": lead,
            "body": body
        }

def main():
    init_db()
    all_news = []

    with get_conn() as connect:
        for ticker in _TICKERS:
            print(f"\n== Fetching news for: {ticker} ==")
            news = fetch_news(ticker)
            all_news.extend(news)
            
            for n in news:
                news_id = insert_news(connect, n)
                print(f"success fetch <news id {news_id}>")
    
    with open('news.json', 'w', encoding='utf-8') as f:
        json.dump(all_news, f,ensure_ascii=False, indent=2)
        
if __name__ == '__main__':
    main()