import pandas as pd

TREND_COLS = {
    "sma": "trend_sma",
    "ema": "trend_ema",
    "adx": "trend_adx",
    "ichimoku": "trend_ichimoku",
}

# funkcja ocenia czy opinia była istotna w kontekście zmiany trendu
def assess_relevance(sentiment, before, after):
    if sentiment == "positive" and before in {"downtrend", "horizontal"} and after == "uptrend":
        return "relevant"
    elif sentiment == "neutral" and before in {"uptrend", "downtrend"} and after == "horizontal":
        return "relevant"
    elif sentiment == "negative" and before in {"uptrend", "horizontal"} and after == "downtrend":
        return "relevant"
    else:
        return "irrelevant"

# funkcja podsumowuje liczbę opinii istotnych i nieistotnych dla każdego wskaźnika analizy technicznej
def summarize_counts(result, methods=None):
    if methods is None:
        methods = list(TREND_COLS.keys())

    frames = []

    for method in methods:
        col = f"relevance_{method}"

        tmp = (result.groupby(["sentiment", col]).size().unstack(fill_value=0))

        for col in ["relevant", "irrelevant"]:
            if col not in tmp.columns:
                tmp[col] = 0

        tmp.columns = pd.MultiIndex.from_product([[method], tmp.columns])
        frames.append(tmp)

    out = pd.concat(frames, axis=1).sort_index()
    return out

# funkcja oblicza procentowy udział opinii istotnych dla każdego wskaźnika analizy technicznej
def summarize_rates(summary):
    rates = {}

    for method in summary.columns.get_level_values(0).unique():
        relevant = summary[(method, "relevant")]
        irrelevant = summary[(method, "irrelevant")]
        total = (relevant + irrelevant).replace(0, pd.NA)
        rates[method] = (relevant / total) * 100

    return pd.DataFrame(rates).round(1)

# funkcja przeprowadza analizę istotności opinii na podstawie wyników analizy zgodności oraz trendów wyznaczonych dla kolejnych dni okna czasowego
def evaluate_relevancy(events_csv, acc_csv, col_id, trend_cols, prefix):
    events = pd.read_csv(events_csv)
    events["session_date"] = pd.to_datetime(events["session_date"], errors="coerce")

    acc = pd.read_csv(acc_csv)
    acc = acc[acc["accuracy"] == "consistent"]
    acc = acc[[col_id, "ticker", "sentiment", "min_change", "time_window"]]

    acc = acc.copy()
    acc["time_window"] = acc["time_window"].astype(int)

    events = events.copy()
    events["day_offset"] = events["day_offset"].astype(int)

    df = acc.copy()
    base0 = (
        events[events["day_offset"] == 0][[col_id, "session_date", "close"]]
        .rename(columns={"session_date": "date_0", "close": "close_0"})
    )
    baseT = (
        events[[col_id, "day_offset", "session_date", "close"]]
        .rename(columns={"day_offset": "time_window", "session_date": "date_T", "close": "close_T"})
    )
    baseT["time_window"] = baseT["time_window"].astype(int)

    df = (
        df.merge(base0, on=col_id, how="left")
          .merge(baseT, on=[col_id, "time_window"], how="left")
    )

    for method, trend_col in trend_cols.items():
        trend0 = (
            events[events["day_offset"] == 0][[col_id, trend_col]]
            .rename(columns={trend_col: f"trend_before_{method}"})
        )
        trendT = (
            events[[col_id, "day_offset", trend_col]]
            .rename(columns={"day_offset": "time_window", trend_col: f"trend_after_{method}"})
        )

        df = (
            df
            .merge(trend0, on=col_id, how="left")
            .merge(trendT, on=[col_id, "time_window"], how="left")
        )

        df[f"relevance_{method}"] = df.apply(
            lambda row: assess_relevance(
                row["sentiment"], row[f"trend_before_{method}"], row[f"trend_after_{method}"]), axis=1)

    cols = [
        col_id,
        "ticker",
        "sentiment",
        "min_change",
        "time_window",
        "date_0",
        "date_T",
        "close_0",
        "close_T"
    ]

    for method in trend_cols.keys():
        cols += [f"trend_before_{method}", f"trend_after_{method}", f"relevance_{method}"]

    result = df[cols].sort_values([col_id, "time_window"]).reset_index(drop=True)
    result.to_csv(f"datasets/relevancy/full_relevant_{prefix}_analysis.csv", index=False)

    return result

def main():
    news_relevancy = evaluate_relevancy(
        events_csv="datasets/events/events_news.csv",
        acc_csv="datasets/consistency/consistent_news.csv",
        col_id="news_id",
        trend_cols=TREND_COLS,
        prefix="news"
    )

    news_relevancy_counts = summarize_counts(news_relevancy)
    news_relevancy_rates = summarize_rates(news_relevancy_counts)
    news_relevancy_counts.to_csv("datasets/relevancy/news_relevancy_counts.csv", index=False)
    news_relevancy_rates.to_csv("datasets/relevancy/news_relevancy_rates.csv", index=False)
    print(news_relevancy)
    print(news_relevancy_counts)
    print(news_relevancy_rates)

    posts_relevancy = evaluate_relevancy(
        events_csv="datasets/events/events_posts.csv",
        acc_csv="datasets/consistency/consistent_posts.csv",
        col_id="post_id",
        trend_cols=TREND_COLS,
        prefix="posts"
    )

    posts_relevancy_counts = summarize_counts(posts_relevancy)
    posts_relevancy_rates = summarize_rates(posts_relevancy_counts)
    posts_relevancy_counts.to_csv("datasets/relevancy/posts_relevancy_counts.csv", index=False)
    posts_relevancy_rates.to_csv("datasets/relevancy/posts_relevancy_rates.csv", index=False)
    print(posts_relevancy)
    print(posts_relevancy_counts)
    print(posts_relevancy_rates)

if __name__ == "__main__":
    main()