import os
import difflib
import argparse
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate

# Inline lightweight data loading and preprocessing to avoid import issues
import os
import re
import unicodedata

def normalize_bangla_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\u200B\u200C\u200D\u180E\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["reference_article", "social_item", "altered_excerpt", "original_excerpt"]:
        if col in out.columns:
            out[col] = out[col].apply(normalize_bangla_text)
    return out

def load_data(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    try:
        df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(data_path, encoding="utf-8", engine="python")
    if "mani_news" in df.columns and "original_news_article" in df.columns:
        df = df.rename(columns={"mani_news": "social_item", "original_news_article": "reference_article"})
    train_df = df[df.get("data_type") == 'TRAIN'].reset_index(drop=True)
    test_df = df[df.get("data_type") == 'TEST'].reset_index(drop=True)
    return df, train_df, test_df


def exact_match(a: str, b: str) -> int:
    if a is None or b is None:
        return 0
    a_clean = normalize_bangla_text(str(a)).strip()
    b_clean = normalize_bangla_text(str(b)).strip()
    return 1 if a_clean == b_clean else 0


def rouge_l_score(preds: List[str], refs: List[str]) -> float:
    rouge = evaluate.load("rouge")
    # rouge expects newline-separated pairs or lists
    results = rouge.compute(predictions=preds, references=refs)
    return results.get("rougeL", 0.0)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def baseline_predictions(test_df: pd.DataFrame, sim_threshold: float = 0.6) -> pd.DataFrame:
    preds = []
    for _, row in test_df.iterrows():
        ref = row.get("reference_article", "")
        social = row.get("social_item", "")
        sim = similarity(ref, social)
        label = 1 if sim < sim_threshold and len(social) > 0 else 0
        # simple span baseline: none
        preds.append({
            "manipulated": "yes" if label == 1 else "no",
            "altered_excerpt": "none",
            "original_excerpt": "none",
        })
    return pd.DataFrame(preds)


def evaluate_subtasks(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    # Subtask 1: classification
    y_true = (test_df['mani_status'] == 'MANI').astype(int).values
    y_pred = (pred_df['manipulated'].str.lower() == 'yes').astype(int).values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Subtask 1 - Manipulation Detection (Classification)")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    # Subtask 2 & 3: span extraction (compute EM and ROUGE-L)
    altered_refs = test_df['altered_excerpt'].fillna('none').astype(str).tolist()
    original_refs = test_df['original_excerpt'].fillna('none').astype(str).tolist()

    altered_preds = pred_df['altered_excerpt'].fillna('none').astype(str).tolist()
    original_preds = pred_df['original_excerpt'].fillna('none').astype(str).tolist()

    # Exact Match
    em_altered = np.mean([exact_match(p, r) for p, r in zip(altered_preds, altered_refs)])
    em_original = np.mean([exact_match(p, r) for p, r in zip(original_preds, original_refs)])

    # ROUGE-L
    rouge_altered = rouge_l_score(altered_preds, altered_refs)
    rouge_original = rouge_l_score(original_preds, original_refs)

    print("\nSubtask 2 - Altered Excerpt Extraction")
    print(f"  Exact Match (EM): {em_altered:.4f}")
    print(f"  ROUGE-L:         {rouge_altered:.4f}")

    print("\nSubtask 3 - Original Excerpt Extraction")
    print(f"  Exact Match (EM): {em_original:.4f}")
    print(f"  ROUGE-L:         {rouge_original:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', help='Path to predictions CSV with columns: manipulated, altered_excerpt, original_excerpt')
    parser.add_argument('--sim-threshold', type=float, default=0.6, help='Similarity threshold for baseline classifier')
    args = parser.parse_args()

    print('Loading dataset...')
    df, train_df, test_df = load_data('Data/BanMANI.csv')
    test_df = preprocess_data(test_df)

    if args.predictions and os.path.exists(args.predictions):
        print(f'Loading predictions from {args.predictions}')
        pred_df = pd.read_csv(args.predictions)
    else:
        print('No predictions file provided. Generating simple baseline predictions...')
        pred_df = baseline_predictions(test_df, sim_threshold=args.sim_threshold)

    evaluate_subtasks(test_df, pred_df)

    # Save baseline predictions for review
    pred_df.to_csv('predictions_baseline.csv', index=False)
    print('\nSaved baseline predictions to predictions_baseline.csv')
