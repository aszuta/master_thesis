import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
POSTS_JSON = BASE_DIR / "datasets" / "json" / "posts_data.json"
FILTERED_CSV = BASE_DIR / "datasets" / "filtered" / "filtered_data.csv"

ID_COL_FILTERED = "id"
TEXT_COL = "body"

PER_CLASS = 100
HALF = PER_CLASS // 2
SEED = 42

with open(POSTS_JSON, "r", encoding="utf-8") as f:
    preds = json.load(f)

preds_df = pd.DataFrame(preds)
posts_df = pd.read_csv(FILTERED_CSV).rename(columns={ID_COL_FILTERED: "post_id"})
df = posts_df.merge(preds_df, on="post_id", how="inner")

# wybiera przykłady z danej klasy sentymentu z uwzględnieniem wartości confidence
def sample_by_confidence(g: pd.DataFrame):
    g = g.dropna(subset=["confidence"])

    if len(g) <= PER_CLASS:
        return g.sample(frac=1, random_state=SEED)
    
    g_sorted = g.sort_values("confidence", ascending=False)
    top = g_sorted.head(HALF)
    bottom = g_sorted.tail(PER_CLASS - HALF)
    out = pd.concat([top, bottom]).sample(frac=1, random_state=SEED)

    return out

sampled = (
    df.groupby("sentiment_label", group_keys=False)
      .apply(sample_by_confidence)
      .reset_index(drop=True)
)

keep_cols = ["post_id", TEXT_COL, "sentiment_label", "confidence",
             "p_positive", "p_neutral", "p_negative", "sentiment_score"]

keep_cols = [c for c in keep_cols if c in sampled.columns]
sampled = sampled[keep_cols].copy()

sampled["manual_label"] = ""
sampled.to_csv(BASE_DIR / "datasets" / "model_accuracy" /"bert_manual_sample_300_mixed.csv", index=False, encoding="utf-8-sig")

print(sampled["sentiment_label"].value_counts())
