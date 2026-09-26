import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

PRICES_DIR = "prices"

trend_colors = {
    "Uptrend": "green",
    "Downtrend": "red",
    "Horizontal": "blue"
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

    max = df["Max."]
    min = df["Min."]
    close = df["Ostatnio"]

    plus_dm = max.diff()
    minus_dm = -min.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([max - min, (max - close.shift()).abs(), (min - close.shift()).abs()], axis=1).max(axis=1)
    tr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    plus_dm_rma = plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_rma = minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    plus_DI = 100 * (plus_dm_rma / tr14)
    minus_DI = 100 * (minus_dm_rma / tr14)

    dx = (abs(plus_DI - minus_DI) / (plus_DI + minus_DI)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

    df["ADX"] = adx
    df["+DI"] = plus_DI
    df["-DI"] = minus_DI

    df["trend"] = "Horizontal"
    mask_trend = df["ADX"] >= 20
    df.loc[mask_trend & (df["+DI"] > df["-DI"]), "trend"] = "Uptrend"
    df.loc[mask_trend & (df["-DI"] > df["+DI"]), "trend"] = "Downtrend"

    df = df.set_index("Data")

    df = df.loc[START_DATE:END_DATE].copy()

    change_points = df.index[df["trend"].ne(df["trend"].shift())]
    segment_starts = list(change_points)
    segment_ends = segment_starts[1:] + [df.index[-1]]

    fig, (ax_price, ax_ind) = plt.subplots(2, 1, figsize=(12, 6))

    ax_price.plot(df.index, df["Ostatnio"], color="black", linewidth=1.8, label="Price")

    ax_price.set_title(f"Trend detection using ADX - {ticker}")
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("Date")
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

    ax_ind.plot(df.index, df["+DI"], label="+DI", color="blue")
    ax_ind.plot(df.index, df["-DI"], label="-DI", color="orange")
    ax_adx = ax_ind.twinx()
    ax_adx.plot(df.index, df["ADX"], label="ADX", color="green", linestyle="--")
    ax_adx.axhline(20, linestyle=":", color="blue", label="ADX = 20")

    ax_adx.set_ylabel("ADX")
    ax_ind.set_ylabel("ADX")
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
    lines2, labels2 = ax_adx.get_legend_handles_labels()
    ax_ind.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()