import pandas as pd
from pathlib import Path

# ocenia zgodność sentymentu opinii z późniejszą zmianą ceny akcji
def evaluate_accuracy(data, col_id, min_change, time_window, label_col):
    day0 = data[data["day_offset"] == 0]
    day_t = data[data["day_offset"] == time_window]

    merged = day0.merge(day_t, on=col_id, suffixes=("_0", f"_{time_window}"))
    merged["return_t"] = (merged[f"close_{time_window}"] / merged["close_0"] - 1) * 100

    def assess(row):
        sentiment = row[f"{label_col}_label_0"]
        ret = row["return_t"]

        if sentiment == "positive" and ret >= min_change:
            return "consistent"
        elif sentiment == "negative" and ret <= -min_change:
            return "consistent"
        elif sentiment == "neutral" and abs(ret) <= min_change:
            return "consistent"
        else:
            return "inconsistent"

    merged["accuracy"] = merged.apply(assess, axis=1)

    result = merged[[col_id, "ticker_0", f"{label_col}_label_0", "return_t", "accuracy"]].rename(columns={
        "ticker_0": "ticker",
        f"{label_col}_label_0": "sentiment", 
        "return_t": f"return_{time_window}",
    })
    
    return result

# funkcja podsumowuje liczbę opinii zgodnych i niezgodnych z reakcją rynku
def summarize_counts(result):
    summary = (
        result.groupby("sentiment")["accuracy"].value_counts().unstack(fill_value=0)
    )

    for col in ["consistent", "inconsistent"]:
        if col not in summary.columns:
            summary[col] = 0

    return summary[["consistent", "inconsistent"]].sort_index()

# wynokune analizę dla różnych kombinacji okna czasowego i minimalnej wartości stopy zwrotu
def grid_search(input_csv, col_id, min_change, time_window, prefix, label_col):
    path = Path("datasets") / input_csv
    data = pd.read_csv(path, sep=",", decimal=".")
    data["session_date"] = pd.to_datetime(data["session_date"], errors="coerce")

    outputs = []
    event_rows = []

    for tw in time_window:
        for mc in min_change:
            result = evaluate_accuracy(
                data,
                col_id,
                min_change=float(mc),
                time_window=int(tw),
                label_col=label_col
            )

            summary = summarize_counts(result)

            row = summary.stack()
            row.index = [f"{sentiment}_{accuracy}" for (sentiment, accuracy) in row.index]
            row = row.to_frame().T
            row.insert(0, "time_window", int(tw))
            row.insert(1, "min_change", mc)

            event_df = result.rename(columns={f"return{tw}": "return_pct"}).copy()

            events = data.copy()
            events["day_offset"] = events["day_offset"].astype(int)

            base0 = (
                events[events["day_offset"] == 0][[col_id, "session_date", "close"]]
                .rename(columns={
                    "session_date": "date_0",
                    "close": "close_0"
                })
            )

            baseT = (
                events[[col_id, "day_offset", "session_date", "close"]]
                .rename(columns={
                    "day_offset": "target_day_offset",
                    "session_date": "date_T",
                    "close": "close_T"
                })
            )

            baseT["target_day_offset"] = baseT["target_day_offset"].astype(int)

            event_df["target_day_offset"] = int(tw)

            event_df = (
                event_df
                .merge(base0, on=col_id, how="left")
                .merge(baseT, on=[col_id, "target_day_offset"], how="left")
            )

            event_df["time_window"] = int(tw)
            event_df["min_change"] = mc

            event_df = event_df.drop(columns=["target_day_offset"])

            outputs.append(row)
            event_rows.append(event_df)

    final = (
        pd.concat(outputs, ignore_index=True)
        .fillna(0)
        .astype({"time_window": int, "min_change": int})
    )

    event_data = pd.concat(event_rows, ignore_index=True)

    cols_first = [
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

    cols_first = [c for c in cols_first if c in event_data.columns]
    other_cols = [c for c in event_data.columns if c not in cols_first]

    event_data = event_data[cols_first + other_cols]

    print(event_data)
    # zapisuje plik csv z pełnymi danymi dla każdej opinii, uwzględniając daty sesji, stopy zwrotu oraz wynik oceny zgodności
    event_data.to_csv(f"datasets/consistency/consistent_{prefix}.csv", index=False)

    return final

def main():
    news_accuracy = grid_search(
        input_csv="events/events_news.csv",
        col_id="news_id",
        time_window=range(1, 6),
        min_change=range(1, 11),
        label_col="sentiment",
        prefix="news"
    )

    print(news_accuracy)
    news_accuracy.to_csv("datasets/consistency/news_consistency_analysis.csv", index=False)

    posts_accuracy = grid_search(
        input_csv="events/events_posts.csv",
        col_id="post_id",
        time_window=range(1, 6),
        min_change=range(1, 11),
        label_col="sentiment",
        prefix="posts"
    )

    print(posts_accuracy)
    posts_accuracy.to_csv("datasets/consistency/posts_consistency_analysis.csv", index=False)

if __name__ == "__main__":
    main()