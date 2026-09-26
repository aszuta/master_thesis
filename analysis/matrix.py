import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

path = Path("datasets")/"events"/"events_posts.csv"
posts = pd.read_csv(path, sep=",", decimal=".")
df = pd.read_excel("datasets/model_accuracy/Book1.xlsx", sheet_name="bert_manual_sample_300_mixed")

merge = posts.merge(df[["post_id", "manual_label"]], on="post_id", how="inner")
try:
    merge.to_csv("datasets/model_accuracy/events_manual_label_300.csv", index=False)
    print("Zapisano")
except Exception as e:
    print("Błąd: ", e)

map_labels = {
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative"
}

y_true = df["manual_label"].astype(str).str.lower().str.strip().replace(map_labels)
y_pred = df["sentiment_label"].astype(str).str.lower().str.strip().replace(map_labels)

labels = ["positive", "neutral", "negative"]

cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)

accuracy = np.trace(cm) / np.sum(cm)
precision = np.diagonal(cm) / np.sum(cm, axis=0)
recall = np.diagonal(cm) / np.sum(cm, axis=1)
f1_score = 2 * (precision * recall) / (precision + recall)

avg_precision = np.mean(precision)
avg_recall = np.mean(recall)
avg_f1_score = np.mean(f1_score)

cm_df = pd.DataFrame(
    cm,
    index=[f"True {l}" for l in labels],
    columns=[f"Pred {l}" for l in labels]
)

print("Accuracy:", accuracy)
print("Precision:", precision.round(2))
print("Recall:", recall.round(2))
print("F1-Score:", f1_score.round(2))
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)
print("Average F1-Score:", avg_f1_score)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True
)

plt.title("Confusion Matrix (Sentiment Classification)")
plt.ylabel("Ground truth")
plt.xlabel("Model prediction")
plt.tight_layout()
plt.show()

actual_labels = ["Positive", "Negative", "Neutral"]

x = np.arange(len(actual_labels))

plt.plot(x, precision, marker="o", label="Precision")
plt.plot(x, recall, marker="o", label="Recall")
plt.plot(x, f1_score, marker="o", label="F1-Score")

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.title('Performance Metrics')
plt.xticks(x, actual_labels)
plt.legend()
plt.grid(True)
                            
plt.show()