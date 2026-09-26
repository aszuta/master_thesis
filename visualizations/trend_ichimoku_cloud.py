import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch

PRICES_DIR = "prices"

trend_colors = {
    "Uptrend": "green",
    "Downtrend": "red",
    "Horizontal": "blue",
}

trend_legend = [
    Patch(facecolor="green", alpha=0.2, label="Uptrend"),
    Patch(facecolor="red", alpha=0.2, label="Downtrend"),
    Patch(facecolor="blue", alpha=0.2, label="Horizontal")
]

START_DATE = pd.Timestamp("2025-05-01")
END_DATE   = pd.Timestamp("2026-01-09")

def ichimoku_cloud(df, high_col="Max.", low_col="Min.", tenkan=9, kijun=26, senkou_b=52):
    high = df[high_col]
    low = df[low_col]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen  = (high.rolling(kijun).max()  + low.rolling(kijun).min())  / 2

    span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

    upper = np.maximum(span_a, span_b)
    lower = np.minimum(span_a, span_b)

    return span_a, span_b, upper, lower

for path in Path(PRICES_DIR).glob("*.csv"):
    ticker = path.stem
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y")
    df = df.sort_values("Data").reset_index(drop=True)

    df["SpanA"], df["SpanB"], df["Cloud_Upper"], df["Cloud_Lower"] = ichimoku_cloud(
        df,
        high_col="Max.",
        low_col="Min."
    )

    df["trend"] = "Horizontal"
    df.loc[df["Ostatnio"] > df["Cloud_Upper"], "trend"] = "Uptrend"
    df.loc[df["Ostatnio"] < df["Cloud_Lower"], "trend"] = "Downtrend"

    df = df.set_index("Data")

    df = df.loc[START_DATE:END_DATE].copy()

    change_points = df.index[df["trend"].ne(df["trend"].shift())]
    segment_starts = list(change_points)
    segment_ends = segment_starts[1:] + [df.index[-1]]

    fig, (ax_price, ax_cloud) = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_price.plot(df.index, df["Ostatnio"], color="black", linewidth=1.8, label="Price")
    ax_price.set_title(f"Trend detection with Ichimoku Cloud – {ticker}")
    ax_price.set_ylabel("Price")
    ax_price.grid(True)

    price_line = ax_price.lines[0]
    ax_price.legend(handles=[price_line] + trend_legend, loc="upper left", frameon=True)

    for i in range(len(df) - 1):
        tr = df["trend"].iloc[i]
        ax_price.axvspan(
            df.index[i],
            df.index[i + 1],
            color=trend_colors.get(tr, "blue"),
            alpha=0.08
        )

    ax_cloud.plot(df.index, df["Ostatnio"], color="black", linewidth=1.4, label="Price")
    ax_cloud.plot(df.index, df["SpanA"], label="Span A", color="green", linewidth=1.5)
    ax_cloud.plot(df.index, df["SpanB"], label="Span B", color="brown", linewidth=1.5)

    ax_cloud.fill_between(
        df.index,
        df["Cloud_Upper"],
        df["Cloud_Lower"],
        color="gray",
        alpha=0.3,
        label="Kumo"
    )

    ax_cloud.set_ylabel("Price")
    ax_cloud.set_xlabel("Date")
    ax_cloud.grid(True)

    for i in range(len(df) - 1):
        tr = df["trend"].iloc[i]
        ax_cloud.axvspan(
            df.index[i],
            df.index[i + 1],
            color=trend_colors.get(tr, "blue"),
            alpha=0.08
        )

    ax_cloud.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
