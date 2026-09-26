import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

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

for path in Path(PRICES_DIR).glob("*.csv"):
    ticker = path.stem
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y")
    df = df.sort_values("Data").reset_index(drop=True)

    eps = 0.005

    df["MA(10)"] = df["Ostatnio"].rolling(window=10).mean()
    df["MA(21)"] = df["Ostatnio"].rolling(window=21).mean()

    diff = (df["MA(10)"] - df["MA(21)"]).abs() / df["MA(21)"]

    df["trend"] = "Horizontal"
    df.loc[(diff > eps) & (df["MA(10)"] > df["MA(21)"]), "trend"] = "Uptrend"
    df.loc[(diff > eps) & (df["MA(10)"] < df["MA(21)"]), "trend"] = "Downtrend"

    df = df.set_index("Data")

    df = df.loc[START_DATE:END_DATE].copy()

    change_points = df.index[df["trend"].ne(df["trend"].shift())]
    segment_starts = list(change_points)
    segment_ends = segment_starts[1:] + [df.index[-1]]

    fig, (ax_price, ax_ind) = plt.subplots(
        2, 1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_price.plot(df.index, df["Ostatnio"], color="black", linewidth=1.8, label="Price")

    ax_price.set_title(f"Trend detection using SMA(10) and SMA(21) – {ticker}")
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

    ax_ind.plot(df.index, df["MA(10)"], label="SMA(10)", color="blue")
    ax_ind.plot(df.index, df["MA(21)"], label="SMA(21)", color="orange")

    ax_ind.set_ylabel("Price")
    ax_ind.set_xlabel("Date")
    ax_ind.grid(True)

    for i in range(len(df) - 1):
        tr = df["trend"].iloc[i]
        ax_ind.axvspan(
            df.index[i],
            df.index[i + 1],
            color=trend_colors.get(tr, "blue"),
            alpha=0.08
        )

    lines1, labels1 = ax_ind.get_legend_handles_labels()
    ax_ind.legend(lines1, labels1, loc="upper left")

    plt.tight_layout()
    plt.show()