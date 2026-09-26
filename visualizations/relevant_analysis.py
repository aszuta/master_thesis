import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# wybrane przypadki do wizualizacji
CASE_IDS = {
    "positive": 15,
    "neutral": 21925,
    "negative": 8675
}

styles = {
    "positive": {"color": "green", "label": "Poz"},
    "neutral":  {"color": "blue",  "label": "Neu"},
    "negative": {"color": "red",   "label": "Neg"},
}

trend_colors = {
    "uptrend": "green",
    "downtrend": "red",
    "horizontal": "blue",
}

trend_legend = [
    Patch(facecolor="green", alpha=0.2, label="Uptrend"),
    Patch(facecolor="red", alpha=0.2, label="Downtrend"),
    Patch(facecolor="blue", alpha=0.2, label="Horizontal")
]

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

# funkcja wyznacza trend z wykorzystaniem wskaźnika ADX
def add_trend_adx(df):
    df = df.copy()

    high = df["Max."]
    low = df["Min."]
    close = df["Ostatnio"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

    tr_n = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_dm_n = plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_n = minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    plus_di = 100 * (plus_dm_n / tr_n)
    minus_di = 100 * (minus_dm_n / tr_n)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    df["+DI"] = plus_di
    df["-DI"] = minus_di
    df["ADX"] = adx

    df["trend"] = "horizontal"
    strong = df["ADX"] >= 20
    df.loc[strong & (df["+DI"] > df["-DI"]), "trend"] = "uptrend"
    df.loc[strong & (df["-DI"] > df["+DI"]), "trend"] = "downtrend"

    return df

# funkcja wybiera okno cenowe wokół daty publikacji opinii
def slice_window(prices, date0, left, right):
    pos0 = prices.index.get_loc(date0)
    start = max(pos0 - left, 0)
    end = min(pos0 + right, len(prices) - 1)

    window = prices.iloc[start:end + 1].copy()

    return window

def add_case_trends_from_prices(case, prices):
    case = case.copy()

    date0 = case["date_0"]
    dateT = case["date_T"]

    if date0 in prices.index:
        case["trend_before_adx"] = str(prices.loc[date0, "trend"])
    else:
        case["trend_before_adx"] = None

    if dateT in prices.index:
        case["trend_after_adx"] = str(prices.loc[dateT, "trend"])
    else:
        case["trend_after_adx"] = None

    return case

# funkcja generuje wykres ceny, wskaźnika ADX oraz oznaczenia trendu
def plot_case_window(window, case, style, adx_threshold=20):
    color = style["color"]
    label = style["label"]

    fig, (ax_price, ax_ind) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_price.plot(window.index, window["Ostatnio"], linewidth=2, color="black", label="Price")

    for i in range(len(window) - 1):
        tr = window["trend"].iloc[i]
        ax_price.axvspan(
            window.index[i],
            window.index[i + 1],
            color=trend_colors.get(tr, "blue"),
            alpha=0.08
        )

    date0 = case["date_0"]
    dateT = case["date_T"]

    ax_price.scatter(date0, case["close_0"], color=color, s=90, zorder=5)
    ax_price.text(date0, case["close_0"], label, color=color, fontsize=11, fontweight="bold", ha="left", va="bottom")

    ax_price.scatter(dateT, case["close_T"], color=color, s=90, zorder=5)

    ax_price.annotate(
        "",
        xy=(dateT, case["close_T"]),
        xytext=(date0, case["close_0"]),
        arrowprops=dict(arrowstyle="->", color=color, lw=2)
    )

    mid_x = date0 + (dateT - date0) / 2
    mid_y = (case["close_0"] + case["close_T"]) / 2
    pct_text = f"{case['return']:+.1f}%"
    ax_price.text(
        mid_x, mid_y, pct_text,
        color=color, fontsize=10, fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85)
    )

    ax_price.set_title(f"CDR relevant analysis with ADX: case — {case['sentiment']}")
    ax_price.set_ylabel("Price")
    ax_price.grid(True)

    ax_price.legend(handles=trend_legend, loc="upper left", frameon=True)

    ax_ind.plot(window.index, window["+DI"], label="+DI", linewidth=1.5)
    ax_ind.plot(window.index, window["-DI"], label="-DI", linewidth=1.5)
    ax_ind.set_ylabel("ADX")
    ax_ind.grid(True)

    for i in range(len(window) - 1):
        tr = window["trend"].iloc[i]
        ax_ind.axvspan(
            window.index[i],
            window.index[i + 1],
            color=trend_colors.get(tr, "blue"),
            alpha=0.08
        )

    ax_adx = ax_ind.twinx()
    ax_adx.plot(window.index, window["ADX"], label="ADX", linestyle="--", linewidth=1.5)
    ax_adx.axhline(adx_threshold, linestyle=":", linewidth=1.2, label=f"ADX = {adx_threshold}")
    ax_adx.set_ylabel("ADX")

    h1, l1 = ax_ind.get_legend_handles_labels()
    h2, l2 = ax_adx.get_legend_handles_labels()
    ax_ind.legend(h1 + h2, l1 + l2, loc="upper left")

    ax_ind.set_xlabel("Date")

    plt.tight_layout()
    plt.show()

def main():
    prices = load_price()
    events = load_events()

    prices = add_trend_adx(prices)

    cases = {k: get_case(events, pid, ANALYSIS_T) for k, pid in CASE_IDS.items()}

    for key in ["positive", "neutral", "negative"]:
        case = cases[key]
        df_window = slice_window(prices, case["date_0"], WINDOW_LEFT, WINDOW_RIGHT)
        case = add_case_trends_from_prices(case, prices)
        plot_case_window(df_window, case, styles[key])

if __name__ == "__main__":
    main()