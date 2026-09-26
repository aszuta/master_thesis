import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TREND_METHODS = ["sma", "ema", "adx", "ichimoku"]
SENTIMENTS = ["positive", "neutral", "negative"]

def relevance_matrix(event_csv: str, methods=TREND_METHODS):
    df = pd.read_csv(event_csv)

    rows = []
    for m in methods:
        col = f"relevance_{m}"
        if col not in df.columns:
            continue

        tmp = (
            df[df[col].isin(["relevant", "irrelevant"])]
            .groupby(["sentiment", col])
            .size()
            .unstack(fill_value=0)
        )

        for c in ["relevant", "irrelevant"]:
            if c not in tmp.columns:
                tmp[c] = 0

        rate = tmp["relevant"] / (tmp["relevant"] + tmp["irrelevant"]).replace(0, np.nan)
        rows.append(rate.rename(m.upper()))

    mat = pd.concat(rows, axis=1)

    mat = mat.reindex(SENTIMENTS)

    return mat

def plot_heatmap(mat: pd.DataFrame, title: str, as_percent=True):
    data = mat.copy()
    if as_percent:
        data = (data * 100)

    plt.figure(figsize=(6, 3))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=0, vmax=100)
    plt.title(title)
    plt.ylabel("Sentiment")
    plt.xlabel("Technical indicator")
    plt.tight_layout()
    plt.show()

mat_news = relevance_matrix("datasets/relevancy/full_relevant_news_analysis.csv")
plot_heatmap(mat_news, "Expert opinions — relevance rate by sentiment and technical indicator", as_percent=True)

mat_posts = relevance_matrix("datasets/relevancy/full_relevant_posts_analysis.csv")
plot_heatmap(mat_posts, "Investor opinions — relevance rate by sentiment and technical indicator", as_percent=True)