import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CASE_IDS = {
    "positive": 15,
    "neutral": 6627,
    "negative": 8605
}

styles = {
    "positive": {"color": "green", "label": "Poz"},
    "neutral":  {"color": "blue",  "label": "Neu"},
    "negative": {"color": "red",   "label": "Neg"},
}

WINDOW_LEFT = 4
WINDOW_RIGHT = 10
ANALYSIS_T = 5

# funkcja wczytuje dane cenowe dla spółki CDR
def load_price():
    path = Path("prices")/"CDR.csv"
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y")
    df = df.sort_values("Data").reset_index(drop=True)
    df = df.set_index("Data")

    return df

# funkcja wczytuje dataset utworzony dla postów
def load_events():
    path = Path("datasets")/"events"/"events_posts.csv"
    events = pd.read_csv(path, sep=",", decimal=".")
    events["session_date"] = pd.to_datetime(events["session_date"])
    events = events[events["ticker"] == "CDR"]

    return events


# funkcja pobiera dane wybranego przypadku po ID posta
def get_case(events, post_id, time_window):
    day0 = events[(events["post_id"] == post_id) & (events["day_offset"] == 0)].iloc[0]
    day5 = events[(events["post_id"] == post_id) & (events["day_offset"] == time_window)].iloc[0]

    close_0 = float(day0["close"])
    close_T = float(day5["close"])
    ret = (close_T / close_0 - 1) * 100

    return {
        "post_id": post_id,
        "sentiment": str(day0["sentiment_label"]),
        "date_0": pd.Timestamp(day0["session_date"]),
        "date_T": pd.Timestamp(day5["session_date"]),
        "close_0": close_0,
        "close_T": close_T,
        "return": ret
    }

# funkcja wybiera okno cenowe wokół daty publikacji opinii
def slice_window(prices, date0, left, right):
    pos0 = prices.index.get_loc(date0)
    start = max(pos0 - left, 0)
    end = min(pos0 + right, len(prices) - 1)

    window = prices.iloc[start:end + 1].copy()

    return window

# funkcja generuje wykres zmiany ceny dla wybranego przypadku
def plot_case_window(window, case, style):
    color = style["color"]
    label = style["label"]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(window.index, window["Ostatnio"], linewidth=2, label="Price")

    date0 = case["date_0"]
    dateT = case["date_T"]

    y0 = case["close_0"]
    ax.scatter(date0, y0, color=color, s=90, zorder=5)
    ax.text(date0, y0, label, color=color, fontsize=11, fontweight="bold", ha="left", va="bottom")
    ax.scatter(dateT, case["close_T"], color=color, s=90, zorder=5)

    ax.annotate(
        "",
        xy=(dateT, case["close_T"]),
        xytext=(date0, case["close_0"]),
        arrowprops=dict(arrowstyle="->", color=color, lw=2)
    )

    mid_x = date0 + (dateT - date0) / 2
    mid_y = (case["close_0"] + case["close_T"]) / 2
    pct_text = f"{case["return"]:+.1f}%"

    ax.text(
        mid_x,
        mid_y,
        pct_text,
        color=color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85)
    )

    ax.set_title(f"CDR naive analysis: case – {case["sentiment"]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    prices = load_price()
    events = load_events()

    cases = {k: get_case(events, pid, ANALYSIS_T) for k, pid in CASE_IDS.items()}

    for key in ["positive", "neutral", "negative"]:
        case = cases[key]
        df_window = slice_window(prices, case["date_0"], WINDOW_LEFT, WINDOW_RIGHT)
        plot_case_window(df_window, case, styles[key])

if __name__ == "__main__":
    main()