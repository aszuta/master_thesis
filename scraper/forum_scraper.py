from bs4 import BeautifulSoup, SoupStrainer
from typing import Dict
import requests
import datetime
import json
from utils.db_utils import init_db, get_conn, insert_thread, insert_post
import random
import time

BANKIER_URL = 'https://www.bankier.pl/forum'
BANKIER_PROFILE_URL = 'https://www.bankier.pl/inwestowanie/profile/quote.html?symbol={}'
BANKIER_THREAD_URL = 'https://www.bankier.pl/forum/{}'
BANKIER_POST_URL = 'https://www.bankier.pl{}'
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/115.0"
]

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def try_get(url, headers=None, retries=20, delay=20):
    for i in range(retries):
        try:
            return requests.get(url, headers=headers, timeout=20)
        except requests.exceptions.RequestException as e:
            print(f"[{i+1}/{retries}] Błąd pobierania: {url} → {e}, retry za {delay}s")
            time.sleep(delay)
    print(f"[BŁĄD] Nieudane pobranie po {retries} próbach: {url}")
    return None

_TICKERS = ['PKN', 'CDR', 'PKO', 'KGH', 'LPP']
_SLUGS: Dict[str, str] = {'PKN': 'orlen', 'CDR': 'cdprojekt', 'PKO': 'pkobp', 'KGH': 'kghm', 'LPP': 'lpp'}

_DATE_FMT = "%Y-%m-%d %H:%M"
DATE_START = datetime.date(2025, 12, 31)
DATE_END = datetime.date(2025, 12, 23)

ONLY_LIST = SoupStrainer("table", class_="threadsList")
ONLY_ANSWER = SoupStrainer('div', id='pageMainContainerRight810')
ONLY_POST = SoupStrainer('ul', class_='threadTree')

# pobiera adres forum Bankier.pl dla danej spółki
def profile_forum(slug):
    profile = BANKIER_PROFILE_URL.format(slug.upper())
    try:
        resp = try_get(profile, headers=get_headers())
    except requests.exceptions.RequestException as e:
        print(f"[BŁĄD] Nie udało się pobrać profilu: {profile} → {e}")
        return None

    time.sleep(random.uniform(1.5, 3))
    soup = BeautifulSoup(resp.text, 'lxml')
    link = soup.select_one('a[href^="https://www.bankier.pl/forum/forum_o_"]')
    if not link:
        return None
    base = link['href'].rstrip('.html')
    return base

# pobiera listę wątków z forum dla wybranego tickera
def fetch_threads(ticker):
    base = profile_forum(_SLUGS[ticker])
    threads = []
    page_num = 1

    while True:
        url = f'{base},{page_num}.html' if page_num > 1 else f'{base}.html'
        print(f"pobieram {url}")

        try:
            resp = try_get(url, headers=get_headers())
        except requests.exceptions.RequestException as e:
            print(f"[BŁĄD] Błąd pobierania strony wątków: {url} → {e}")
            break

        time.sleep(random.uniform(1.5, 3))
        soup = BeautifulSoup(resp.text, 'lxml', parse_only=ONLY_LIST)
        rows = soup.find_all('tr')
        if not rows:
            break

        print(f"Znaleziono: {len(rows)}")

        for row in rows:
            title_cell = row.find('td', class_='threadTitle')
            date_cell = row.find('td', class_='createDate')

            a_tag = title_cell.find('a')
            link = a_tag.get('href')
            if link.startswith('https'):
                continue

            title = a_tag.get_text(strip=True)
            date = date_cell.get_text(strip=True)

            try:
                post_datetime = datetime.datetime.strptime(date, _DATE_FMT)
                post_date = post_datetime.date()

                if post_date > DATE_START:
                    continue

                if post_date < DATE_END:
                    return threads
            except ValueError:
                continue

            threads.append({
                'ticker': ticker,
                'url': BANKIER_THREAD_URL.format(link),
                'title': title,
                'timestamp': post_datetime.isoformat(sep=' ')
            })
        
        page_num += 1
            
    print(f"liczba wątków: {len(threads)}")
    return threads

# pobiera posty z wybranego wątku
def crawl_thread_posts(thread):
    posts = []
    thread_url = thread['url']

    print(f'pobieram stronę wątku: {thread_url}')
    try:
        resp = try_get(thread_url, headers=get_headers())
    except requests.exceptions.RequestException as e:
        print(f"[BŁĄD] Nie udało się pobrać wątku: {thread_url} → {e}")
        return posts

    time.sleep(random.uniform(1.5, 3))
    soup = BeautifulSoup(resp.text, 'lxml', parse_only=ONLY_ANSWER)

    show_all_link = soup.find('a', id='showAllThread')
    if show_all_link and show_all_link.get('href'):
        href = show_all_link.get('href')
        thread_id = None

        if "thread_id=" in href:
            try:
                thread_id = href.split("thread_id=")[1].split("&")[0]
            except IndexError:
                pass

        if thread_id:
            page = 1
            while True:
                paged_url = f"https://www.bankier.pl/forum/pokaz-tresc?thread_id={thread_id}&strona={page}"
                print(f"→ pobieram: {paged_url}")

                try:
                    resp = try_get(paged_url, headers=get_headers())
                except requests.exceptions.RequestException as e:
                    print(f"[BŁĄD] Nie udało się pobrać strony {page}: {paged_url} → {e}")
                    break

                time.sleep(random.uniform(1.5, 3))
                soup = BeautifulSoup(resp.text, 'lxml', parse_only=ONLY_POST)

                page_posts = parse_thread_listing(soup, thread)
                if not page_posts:
                    break

                posts.extend(page_posts)
                page += 1

            return posts

    post = parse_post(soup, thread_url)
    if post:
        posts.append(post)

    return posts

# wyodrębnia dane z pojedynczego posta ze strony wątku
def parse_post(soup, url):
    meta = soup.select_one('div.boxMeta')
    author_span = meta.find('span', class_='author').get_text(strip=True)
    author = author_span.removeprefix('Author: ~')
    time = meta.find('time', class_='entry-date').get_text(strip=True)
    title = soup.select_one('div.boxHeader h1').get_text(strip=True)
    body = soup.select_one('div.boxContent div').get_text(' ', strip=True)
    return {'url': url, 'title': title, 'author': author, 'timestamp': time, 'body': body}

# wyodrębnia dane z listy postów w rozwiniętym wątku
def parse_thread_listing(soup, thread):
    posts = []

    for li in soup.select('ul.threadTree li'):
        if not li.find('div', class_='p'):
            continue
        a = li.find('a')
        if not a:
            continue

        href = a.get('href')
        if not href or href.startswith('https'):
            continue

        text = a.get_text(strip=True)
        author_span = li.find("span", class_="author").get_text(strip=True)
        author = author_span.removeprefix('Author: ~')
        date_span = li.find("time", class_="entry-date")
        date = date_span.get_text(strip=True) if date_span else None
        body_div = li.find('div', class_='p')
        body = body_div.get_text(strip=True) if body_div else None

        posts.append({
            'ticker': thread['ticker'],
            'thread_url': thread['url'],
            'url': BANKIER_POST_URL.format(href), 
            'title': text, 
            'author': author if author_span else None, 
            'timestamp': date, 
            'body': body
        })

    return posts

def main():
    init_db()
    all_threads = []
    all_posts = []

    for ticker in _TICKERS:
        threads = fetch_threads(ticker)
        all_threads.extend(threads)

        for thread in threads:
            print(f"\n== Wątek: {thread['title']} ==")
            posts = crawl_thread_posts(thread)
            all_posts.extend(posts)

            with get_conn() as connect:
                thread_id = insert_thread(connect, thread)

                for post in posts:
                    post_id = insert_post(connect, post, thread_id)
                    print(f"success fetch <post id {post_id}> from {thread_id}")
    
    with open('threads.json', 'w', encoding='utf-8') as f:
        json.dump(all_threads, f, ensure_ascii=False, indent=2)

    with open('posts.json', 'w', encoding='utf-8') as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()