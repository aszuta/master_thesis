import pandas as pd
import re
from pathlib import Path

JSON_FILE = Path('cleaned_posts.json')
KEYWORDS_FILE = Path('keywords.txt')

# wczytuje słowa kluczowe z pliku tekstowego
def load_keywords(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    keywords = []

    for line in lines:
        line = line.strip()
        keywords.append(line)

    if keywords:
        return keywords

# funkcja filtruje posty na podstawie występowania słów kluczowych i zapisuje wynik do pliku
def main():
    df = pd.read_json(JSON_FILE, convert_dates=False)
    print(len(df))
    KEYWORDS = load_keywords(KEYWORDS_FILE)

    pattern = r'\b(?:' + '|'.join(re.escape(k) + r'\w*' for k in KEYWORDS) + r')\b'
    PATTERN = re.compile(pattern, flags=re.I)             

    mask = df['body'].str.contains(PATTERN, na=False)
    df_filtered = df[mask].copy()

    print(f"Znalezionych postów: {mask.sum()} z {len(df)}")
    print(df_filtered[['ticker','author','timestamp','title','body']].head(20))

    filtered_data = df_filtered.to_json(orient='records', force_ascii=False, indent=2).replace('\\/', '/')
    with open('datasets/filtered/filtered_data.json', 'w', encoding='utf-8') as f:
        f.write(filtered_data)

    df_filtered.to_csv("datasets/filtered/filtered_data.csv", index=False)

if __name__ == '__main__':  
    main()