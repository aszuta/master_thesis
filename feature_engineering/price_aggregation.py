import sqlite3
import pandas as pd
from pathlib import Path
import datetime as dt
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from analysis.sentiment_HerBERT import sentiment_score

# funkcja ładująca ceny akcji spółek
def load_prices(PRICES_DIR, ticker):
    path = Path(PRICES_DIR)/f"{ticker}.csv"
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y")
    df = df.sort_values("Data").reset_index(drop=True)
    return df

# detekcja trendu z wykorzystaniem SMA
def add_trend_SMA(df, trend_col):
    df = df.copy()

    df["MA(10)"] = df["Ostatnio"].rolling(window=10).mean()
    df["MA(21)"] = df["Ostatnio"].rolling(window=21).mean()
    df[["MA(10)", "MA(21)"]] = df[["MA(10)", "MA(21)"]].round(2)

    diff = (df["MA(10)"] - df["MA(21)"]).abs() / df["MA(21)"]
    eps = 0.005

    df[trend_col] = "horizontal"
    df.loc[(diff > eps) & (df["MA(10)"] > df["MA(21)"]), trend_col] = "uptrend"
    df.loc[(diff > eps) & (df["MA(10)"] < df["MA(21)"]), trend_col] = "downtrend"

    return df

#detekcja trendu z wykorzystaniem EMA
def add_trend_EMA(df, trend_col):
    df = df.copy()

    def ema_from_sma(close, window):
        alpha = 2 / (window + 1)
        ema = pd.Series(np.nan, index=close.index, dtype="float64")
        sma = close.rolling(window=window).mean()

        first_idx = window - 1
        if len(close) <= first_idx:
            return ema
        
        ema.iloc[first_idx] = sma.iloc[first_idx]
        for i in range(first_idx + 1, len(close)):
            ema.iloc[i] = alpha * close.iloc[i] + (1 - alpha) * ema.iloc[i - 1]

        return ema
    
    eps = 0.005

    df["EMA(10)"] = ema_from_sma(df["Ostatnio"], window=10)
    df["EMA(21)"] = ema_from_sma(df["Ostatnio"], window=21)

    diff = (df["EMA(10)"] - df["EMA(21)"]).abs() / df["EMA(21)"]

    df[trend_col] = "horizontal"
    df.loc[(diff > eps) & (df["EMA(10)"] > df["EMA(21)"]), trend_col] = "uptrend"
    df.loc[(diff > eps) & (df["EMA(10)"] < df["EMA(21)"]), trend_col] = "downtrend"

    return df

# detekcja trendu z wykorzystaniem ADX
def add_trend_ADX(df, trend_col):
    df = df.copy()

    max = df["Max."]
    min = df["Min."]
    close = df["Ostatnio"]

    plus_dm = max.diff()
    minus_dm = -min.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([max - min, (max - close.shift()).abs(), (min - close.shift()).abs()], axis=1).max(axis=1)
    tr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_rma  = plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_rma = minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    plus_di = 100 * (plus_dm_rma / tr14)
    minus_di = 100 * (minus_dm_rma / tr14)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    df[trend_col] = "horizontal"
    df.loc[(adx >= 20) & (plus_di > minus_di), trend_col] = "uptrend"
    df.loc[(adx >= 20) & (minus_di > plus_di), trend_col] = "downtrend"

    return df

# detekcja trendu z wykorzystaniem Ichimoku Cloud
def add_trend_ichimoku(df, trend_col):
    df = df.copy()

    high = df["Max."]
    low = df["Min."]

    tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    upper = np.maximum(span_a, span_b)
    lower = np.minimum(span_a, span_b)

    df[trend_col] = "horizontal"
    df.loc[df["Ostatnio"] > upper, trend_col] = "uptrend"
    df.loc[df["Ostatnio"] < lower, trend_col] = "downtrend"

    return df

# funkcja przypisująca odpowiednią datę sesyjną
def assign_session_date(timestamp):
    return timestamp.where(
        timestamp.dt.time < dt.time(17, 0),
        timestamp + pd.Timedelta(days=1)
    ).dt.normalize()

# funkcja przypisująca daty sesyjne do dni okna
def get_window_sessions(prices, session_date, n_sessions):
    window = prices[prices["Data"] >= session_date].head(n_sessions)

    if len(window) < n_sessions:
        return None
    
    return window.copy()

# funkcja obliczająca dzienne stopy zwrotu
def daily_returns(news_df, id_column_name):
    news_df = news_df.sort_values([id_column_name, "day_offset"]).reset_index(drop=True)
    news_df["ret_1d"] = news_df.groupby(id_column_name)["close"].pct_change()
    return news_df

# funkcja tworząca okienko czasowe
def build_news_window(news, PRICES_DIR, id_column_name):
    prices_cache: dict[str, pd.DataFrame] = {}
    rows = []

    for _, n in news.iterrows():
        ticker = n["ticker"]
        pub = n["timestamp"]

        pub_session = assign_session_date(pd.Series([pub])).iloc[0]

        if ticker not in prices_cache:
            dfp = load_prices(PRICES_DIR, ticker)
            dfp = add_trend_SMA(dfp, trend_col="trend_SMA")
            dfp = add_trend_EMA(dfp, trend_col="trend_EMA")
            dfp = add_trend_ADX(dfp, trend_col="trend_ADX")
            dfp = add_trend_ichimoku(dfp, trend_col="trend_ichimoku")
            prices_cache[ticker] = dfp

        prices = prices_cache[ticker]
        window = get_window_sessions(prices, pub_session, 6)

        if window is None: 
            continue

        for offset, (_, w) in enumerate(window.iterrows()):
            rows.append(
                {
                    id_column_name: n["id"], 
                    "ticker": ticker,
                    "published_at": pub,
                    "day_offset": offset,
                    "session_date": w["Data"],
                    "close": w["Ostatnio"],
                    "trend_sma": w["trend_SMA"],
                    "trend_ema": w["trend_EMA"],
                    "trend_adx": w["trend_ADX"],
                    "trend_ichimoku": w["trend_ichimoku"]
                }
            )
        
    return pd.DataFrame(rows)

# funkcja pobierająca dane tekstowe z bazy danych
def load_data(DB_PATH, SQL):
    connect = sqlite3.connect(DB_PATH)
    try:
        data = pd.read_sql_query(SQL, connect)
    finally:
        connect.close()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data

def main():
    DB_PATH = BASE_DIR / "database/forum.db"
    PRICES_DIR = BASE_DIR / "prices"

    NEWS_SQL = """
        SELECT
            id,
            ticker,
            timestamp,
            title,
            full_text
        FROM news
    """

    POSTS_SQL = """
        SELECT
            id,
            ticker,
            timestamp,
            title,
            body
        FROM cleaned_posts
    """

    # news = load_data(DB_PATH, NEWS_SQL)
    posts = load_data(DB_PATH, POSTS_SQL)

    #HerBERT sentiment
    # sent = sentiment_score(
    #     texts=news["full_text"].astype(str).tolist(),
    #     ids=news["id"].tolist(),
    #     batch_size=16
    # )

    sent = sentiment_score(
        texts=posts["body"].astype(str).tolist(),
        ids=posts["id"].tolist(),
        batch_size=16
    )

    # n = build_news_window(news, PRICES_DIR, id_column_name="news_id")
    # n = daily_returns(n, id_column_name="news_id")

    p = build_news_window(posts, PRICES_DIR, id_column_name="post_id")
    p = daily_returns(p, id_column_name="post_id")

    # n = n.merge(sent, on="news_id", how="left")
    # n = n.drop(columns=["p_negative", "p_neutral", "p_positive", "confidence"], errors="ignore")
    # n.to_csv(BASE_DIR / "datasets" / "events" / "events_news.csv", index=False)
    
    p = p.merge(sent, on="post_id", how="left")
    p = p.drop(columns=["p_negative", "p_neutral", "p_positive", "confidence"], errors="ignore")
    p.to_csv(BASE_DIR / "datasets" / "events" / "events_posts.csv", index=False)

    # print(n)
    print(p)

if __name__ == "__main__":
    main()