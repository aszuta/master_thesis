import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import json

# mapowanie etykiet zwracanych przez model na nazwy klas sentymentu
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# wczytywanie tokenizera i modelu HerBERT
tokenizer = AutoTokenizer.from_pretrained("Voicelab/herbert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("Voicelab/herbert-base-cased-sentiment")
model.eval()

# oblicza sentyment dla listy tekstów
@torch.no_grad()
def sentiment_score(texts, ids, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)

        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for nid, p in zip(batch_ids, probs):
            p_neg, p_neu, p_pos = float(p[0]), float(p[1]), float(p[2])
            label_id = int(np.argmax(p))
            rows.append({
                "post_id": nid,
                "sentiment_label": id2label[label_id],
                "p_negative": p_neg,
                "p_neutral": p_neu,
                "p_positive": p_pos,
                "sentiment_score": p_pos - p_neg,
                "confidence": max(p_neg, p_neu, p_pos),
            })

    with open("posts_data.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return pd.DataFrame(rows)