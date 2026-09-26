import sqlite3
from pathlib import Path
from contextlib import contextmanager

# ścieżka do lokalnej bazy danych w sqlite
_DB_PATH = Path(__file__).resolve().parents[1] / "database" / "forum.db"

# definicja struktury tabel
SCHEMA = """
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        url TEXT UNIQUE,
        title TEXT,
        timestamp TEXT
    );

    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER REFERENCES threads(id) ON DELETE CASCADE,
        url TEXT UNIQUE,
        title TEXT,
        author TEXT,
        timestamp TEXT,
        body TEXT
    );

    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        ticker TEXT,
        timestamp TEXT,
        title TEXT,
        lead TEXT,
        full_text TEXT
    );

    CREATE TABLE IF NOT EXISTS cleaned_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT,
        url TEXT UNIQUE,
        title TEXT,
        author TEXT,
        timestamp TEXT,
        body TEXT
    );
"""

# otwiera połączenie z bazą danych
@contextmanager
def get_conn():
    connect = sqlite3.connect(_DB_PATH)
    connect.execute("PRAGMA foreign_keys = ON;")
    try:
        yield connect
        connect.commit()
    finally:
        connect.close()

# inicjalizuje bazę danych i tworzy wymagane tabele
def init_db():
    with get_conn() as connect:
        connect.executescript(SCHEMA)

# zapisuje wątek forum w bazie danych lub zwraca nr ID już istniejącego
def insert_thread(connect, thread):
    cursor = connect.execute(
        "INSERT OR IGNORE INTO threads (ticker, url, title, timestamp) VALUES (?, ?, ?, ?)",
        (thread["ticker"], thread["url"], thread["title"], thread["timestamp"])
    )

    if cursor.lastrowid:
        return cursor.lastrowid
    
    cursor = connect.execute("SELECT id FROM threads WHERE url = ?", (thread["url"],))
    return cursor.fetchone()[0]

# zapisuje post z forum w bazie danych lub zwraca nr ID już istniejącego
def insert_post(connect, post, thread_id):
    cursor = connect.execute(
        "INSERT OR IGNORE INTO posts (thread_id, url, title, author, timestamp, body) VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, post["url"], post["title"], post["author"], post["timestamp"], post["body"])
    )

    if cursor.lastrowid:
        return cursor.lastrowid
    
    cursor = connect.execute("SELECT id FROM posts WHERE url = ?", (post["url"],))
    return cursor.fetchone()[0]

# zapisuje wiadomość z sekcji wiadomości w bazie danych lub zwraca nr ID już istniejącego
def insert_news(connect, news):
    cursor = connect.execute(
        "INSERT OR IGNORE INTO news (url, ticker, timestamp, title, lead, full_text) VALUES (?, ?, ?, ?, ?, ?)",
        (news["url"], news["ticker"], news["timestamp"], news["title"], news["lead"], news["full_text"])
    )

    if cursor.lastrowid:
        return cursor.lastrowid
    
    cursor = connect.execute("SELECT id FROM news WHERE url = ?", (news["url"],))
    return cursor.fetchone()[0]

# zapisuje w bazie danych oczyszczone posty z forum
def insert_cleaned_posts(connect, posts):
    rows = list(posts[["ticker", "title", "author", "timestamp", "body"]].itertuples(index=False, name=None))
    connect.executemany(
        "INSERT OR IGNORE INTO cleaned_posts (ticker, title, author, timestamp, body) VALUES (?, ?, ?, ?, ?)",
        rows
    )