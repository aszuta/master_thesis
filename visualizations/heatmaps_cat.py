import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_categories(df, bins, labels):
    df = df.copy()
    df["min_change_cat"] = pd.cut(df["min_change"], bins=bins, labels=labels, include_lowest=True)

    return df

def aggredate_by_cat(df):
    df = df.copy()

    agg_dict = {}
    for sentiment in ["positive", "neutral", "negative"]:
        agg_dict[f"{sentiment}_consistent"] = "sum"
        agg_dict[f"{sentiment}_inconsistent"] = "sum"

    group = df.groupby(["time_window", "min_change_cat"], dropna=False).agg(agg_dict).reset_index()

    for sentiment in ["positive", "neutral", "negative"]:
        cons = group[f"{sentiment}_consistent"]
        inco = group[f"{sentiment}_inconsistent"]
        total = cons + inco
        group[f"{sentiment}_rate"] = np.where(total > 0, cons / total, np.nan)

    return group

def generate_heatmap(df, value_col, title):
    pivot = df.pivot(index="time_window", columns="min_change_cat", values=value_col)
    pivot = pivot.sort_index(ascending=False).sort_index(axis=1)
    data = pivot * 100

    labels = data.map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    plt.figure(figsize=(7, 4))

    ax = sns.heatmap(data, annot=labels, fmt="", cmap="coolwarm", cbar=True, vmin=0, vmax=100)

    cbar = ax.collections[0].colorbar
    cbar.set_label("Consistent rate (%)")
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    plt.title(title)
    plt.xlabel("Min change of rate of return")
    plt.ylabel("Time window (days)")

    plt.tight_layout()
    plt.show()

def make_all_heatmaps(grid_csv: str, label: str):
    df = pd.read_csv(grid_csv)
    df = add_categories(df, bins=(1, 2, 5, 10), labels=("slight (1%-2%)", "moderate (3%-5%)", "strong (6%+)"))
    df = aggredate_by_cat(df)

    generate_heatmap(df, "positive_rate", f"{label} — Positive sentiment")
    generate_heatmap(df, "negative_rate", f"{label} — Negative sentiment")
    generate_heatmap(df, "neutral_rate",  f"{label} — Neutral sentiment")

make_all_heatmaps("datasets/consistency/news_consistency_analysis.csv", "Expert opinion")
make_all_heatmaps("datasets/consistency/posts_consistency_analysis.csv", "Investor opinion")