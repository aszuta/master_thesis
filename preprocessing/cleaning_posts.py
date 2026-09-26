import pandas as pd
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from utils.db_utils import init_db, get_conn, insert_cleaned_posts

# ścieżka do bazy danych
DB_PATH = BASE_DIR / "database" / "forum.db"

# zapytanie pobierające posty z forum wraz z tickerem spółki
SQL = """
        SELECT
            posts.id,
            threads.ticker,
            posts.timestamp,
            posts.title,
            posts.author,
            posts.body
        FROM posts
        JOIN threads ON posts.thread_id = threads.id
        WHERE posts.timestamp >= '2025-06-01'
    """

# wyrażenie regularne wykorzystywane do usuwania emotikonów z tekstu
EMOJI_RE = re.compile(
    r'[' 
    r'\U0001F600-\U0001F64F'
    r'\U0001F300-\U0001F5FF'
    r'\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF'
    r'\U00002700-\U000027BF'
    r'\U00002600-\U000026FF'
    r'\U00002B00-\U00002BFF'
    r'\U0001F900-\U0001F9FF'
    r']+',
    flags=re.UNICODE,
)

# funkcja oczyszczające posty za pomocą wyrażeń regularnych
def clean_posts(DB_PATH, SQL):
    connect = sqlite3.connect(DB_PATH)
    try:
        posts = pd.read_sql_query(SQL, connect)
    finally:
        connect.close()

    data = posts.sort_values("timestamp").reset_index(drop=True)

    df = pd.DataFrame(data)
    df["author"] = df["author"].str.replace(r'Autor\s*:?\s*~?', '', regex=True).str.strip()

    df["body"] = df["body"].str.replace(r'https?://\S+', '', regex=True)
    df["body"] = df["body"].apply(lambda text: EMOJI_RE.sub('', text) if isinstance(text, str) else text)
    df["body"] = df["body"].str.replace(r'Dnia .*? napisał\(a\):', '', regex=True)
    df["body"] = df["body"].str.replace(r'(?m)^>.*$', '', regex=True)
    df["body"] = df["body"].str.replace(r'\n+', '\n', regex=True).str.strip()
    df["body"] = df["body"].str.replace(r'\s+', ' ', regex=True).str.strip()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    df["body"] = df["body"].astype(str).str.strip()
    df = df[df["body"].notna() & (df["body"] != "")]

    return df

def main():
    init_db()

    cleaned_posts = clean_posts(DB_PATH, SQL)

    with get_conn() as connect:
        insert_cleaned_posts(connect, cleaned_posts)
        connect.commit()

    print(f"Inserted cleaned posts: {len(cleaned_posts)}")

    cleaned = cleaned_posts.to_json(orient="records", force_ascii=False, indent=2)
    with open("datasets/json/cleaned_posts.json", "w", encoding="utf-8") as f:
        f.write(cleaned)

if __name__ == "__main__":
    main()