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

def ema_from_sma(series, window):
    alpha = 2 / (window + 1)

    ema = pd.Series(np.nan, index=series.index, dtype="float64")
    sma = series.rolling(window=window).mean()

    first_idx = window - 1
    if len(series) <= first_idx:
        return ema 

    ema.iloc[first_idx] = sma.iloc[first_idx]

    for i in range(first_idx + 1, len(series)):
        ema.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * ema.iloc[i - 1]

    return ema

for path in Path(PRICES_DIR).glob("*.csv"):
    ticker = path.stem
    df = pd.read_csv(path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()
    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y")
    df = df.sort_values("Data").reset_index(drop=True)

    eps = 0.005

    df["EMA(10)"] = ema_from_sma(df["Ostatnio"], window=10)
    df["EMA(21)"] = ema_from_sma(df["Ostatnio"], window=21)

    diff = (df["EMA(10)"] - df["EMA(21)"]).abs() / df["EMA(21)"]

    df["trend"] = "Horizontal"
    df.loc[(diff > eps) & (df["EMA(10)"] > df["EMA(21)"]), "trend"] = "Uptrend"
    df.loc[(diff > eps) & (df["EMA(10)"] < df["EMA(21)"]), "trend"] = "Downtrend"

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
    ax_price.set_title(f"Trend detection using EMA(10) and EMA(21) – {ticker}")
    ax_price.set_ylabel("Cena")
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

    ax_ind.plot(df.index, df["EMA(10)"], label="EMA(10)", color="blue")
    ax_ind.plot(df.index, df["EMA(21)"], label="EMA(21)", color="orange")

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

    ax_ind.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
