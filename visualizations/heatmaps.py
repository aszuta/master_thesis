import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_rates(df):
    df = df.copy()

    for sentiment in ["positive", "neutral", "negative"]:
        consistent = df[f"{sentiment}_consistent"]
        inconsistent = df[f"{sentiment}_inconsistent"]
        total = consistent + inconsistent

        df[f"{sentiment}_rate"] = np.where(total > 0, consistent / total, np.nan)

    return df

def generate_heatmap(df, value_col, title):
    pivot = df.pivot(index="time_window", columns="min_change", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 3))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="coolwarm")

    cbar = plt.colorbar(im)
    cbar.set_label("Consistent rate (%)")
    ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])

    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    plt.xlabel("Min change (%)")
    plt.ylabel("Time window (days)")
    plt.title(title)

    plt.tight_layout()
    plt.show()

def make_all_heatmaps(grid_csv: str, label: str):
    df = pd.read_csv(grid_csv)
    df = add_rates(df)

    generate_heatmap(df, "positive_rate", f"{label} — Positive sentiment")
    generate_heatmap(df, "negative_rate", f"{label} — Negative sentiment")
    generate_heatmap(df, "neutral_rate",  f"{label} — Neutral sentiment")
    
make_all_heatmaps("datasets/consistency/news_consistency_analysis.csv", "Expert opinion")
make_all_heatmaps("datasets/consistency/posts_consistency_analysis.csv", "Investor opinion")