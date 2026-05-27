from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.config import (
    FINBERT_LOCAL_PATH,
    FINBERT_MODEL_NAME,
    MAX_TEXT_LENGTH,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_CONFIDENCE_FLOOR,
    USE_LOCAL_FINBERT,
)
from src.utils import clean_text, clamp, logger, safe_float, safe_str


LABEL_NORMALIZATION = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "label_0": "positive",
    "label_1": "negative",
    "label_2": "neutral",
    "0": "positive",
    "1": "negative",
    "2": "neutral",
}


def _get_model_source() -> str:
    if USE_LOCAL_FINBERT and FINBERT_LOCAL_PATH.exists():
        return str(FINBERT_LOCAL_PATH)

    return FINBERT_MODEL_NAME


def _get_device() -> int:
    if torch.cuda.is_available():
        return 0

    return -1


@lru_cache(maxsize=1)
def load_finbert():
    model_source = _get_model_source()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(model_source)

        sentiment_pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device=_get_device(),
            return_all_scores=True,
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
        )

        logger.info("Loaded FinBERT model from %s", model_source)
        return sentiment_pipeline

    except Exception as error:
        logger.exception("Failed to load FinBERT model: %s", error)
        return None


def normalize_label(label: Any) -> str:
    label = safe_str(label).lower().strip()
    return LABEL_NORMALIZATION.get(label, label)


def _empty_prediction(text: str = "") -> Dict[str, Any]:
    return {
        "text": clean_text(text),
        "label": "neutral",
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "positive_prob": 0.0,
        "negative_prob": 0.0,
        "neutral_prob": 1.0,
        "model_available": False,
        "error": None,
    }


def _scores_to_prediction(text: str, scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    probs = {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }

    for item in scores:
        label = normalize_label(item.get("label"))
        score = safe_float(item.get("score"), 0.0)

        if label in probs:
            probs[label] = score

    positive_prob = probs["positive"]
    negative_prob = probs["negative"]
    neutral_prob = probs["neutral"]

    sentiment_score = positive_prob - negative_prob
    sentiment_score = clamp(sentiment_score, -1.0, 1.0)

    label = max(probs, key=probs.get)
    confidence = probs[label]

    if confidence < SENTIMENT_CONFIDENCE_FLOOR:
        label = "neutral"

    return {
        "text": clean_text(text),
        "label": label,
        "sentiment_score": round(float(sentiment_score), 6),
        "confidence": round(float(confidence), 6),
        "positive_prob": round(float(positive_prob), 6),
        "negative_prob": round(float(negative_prob), 6),
        "neutral_prob": round(float(neutral_prob), 6),
        "model_available": True,
        "error": None,
    }


def predict_sentiment(text: Any) -> Dict[str, Any]:
    text = clean_text(text)

    if not text:
        result = _empty_prediction("")
        result["error"] = "Empty text"
        return result

    sentiment_pipeline = load_finbert()

    if sentiment_pipeline is None:
        result = _empty_prediction(text)
        result["error"] = "FinBERT model could not be loaded"
        return result

    try:
        output = sentiment_pipeline(text)

        if isinstance(output, list) and output and isinstance(output[0], list):
            scores = output[0]
        elif isinstance(output, list):
            scores = output
        else:
            scores = []

        return _scores_to_prediction(text, scores)

    except Exception as error:
        logger.exception("Sentiment prediction failed: %s", error)
        result = _empty_prediction(text)
        result["error"] = str(error)
        return result


def analyze_news_batch(news_items: Sequence[Any]) -> pd.DataFrame:
    cleaned_items = [clean_text(item) for item in news_items if clean_text(item)]

    if not cleaned_items:
        return pd.DataFrame(
            columns=[
                "text",
                "label",
                "sentiment_score",
                "confidence",
                "positive_prob",
                "negative_prob",
                "neutral_prob",
                "model_available",
                "error",
            ]
        )

    sentiment_pipeline = load_finbert()

    if sentiment_pipeline is None:
        rows = []

        for text in cleaned_items:
            result = _empty_prediction(text)
            result["error"] = "FinBERT model could not be loaded"
            rows.append(result)

        return pd.DataFrame(rows)

    rows = []

    try:
        for start in range(0, len(cleaned_items), SENTIMENT_BATCH_SIZE):
            batch = cleaned_items[start : start + SENTIMENT_BATCH_SIZE]

            outputs = sentiment_pipeline(batch)

            for text, output in zip(batch, outputs):
                if isinstance(output, list):
                    rows.append(_scores_to_prediction(text, output))
                else:
                    result = _empty_prediction(text)
                    result["error"] = "Invalid model output"
                    rows.append(result)

    except Exception as error:
        logger.exception("Batch sentiment prediction failed: %s", error)

        rows = []
        for text in cleaned_items:
            rows.append(predict_sentiment(text))

    return pd.DataFrame(rows)


def aggregate_sentiment(
    sentiment_df: Optional[pd.DataFrame],
    score_column: str = "sentiment_score",
    confidence_column: str = "confidence",
) -> Dict[str, Any]:
    if sentiment_df is None or sentiment_df.empty:
        return {
            "sentiment_score": 0.0,
            "average_confidence": 0.0,
            "news_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 1.0,
            "dominant_label": "neutral",
        }

    df = sentiment_df.copy()

    if score_column not in df.columns:
        df[score_column] = 0.0

    if confidence_column not in df.columns:
        df[confidence_column] = 0.0

    df[score_column] = pd.to_numeric(df[score_column], errors="coerce").fillna(0.0)
    df[confidence_column] = pd.to_numeric(df[confidence_column], errors="coerce").fillna(0.0)

    weights = df[confidence_column].clip(lower=0.05)

    if weights.sum() == 0:
        sentiment_score = float(df[score_column].mean())
    else:
        sentiment_score = float(np.average(df[score_column], weights=weights))

    sentiment_score = clamp(sentiment_score, -1.0, 1.0)

    labels = df.get("label", pd.Series(["neutral"] * len(df))).fillna("neutral").astype(str)

    positive_count = int((labels == "positive").sum())
    negative_count = int((labels == "negative").sum())
    neutral_count = int((labels == "neutral").sum())
    news_count = int(len(df))

    label_counts = {
        "positive": positive_count,
        "negative": negative_count,
        "neutral": neutral_count,
    }

    dominant_label = max(label_counts, key=label_counts.get)

    return {
        "sentiment_score": round(float(sentiment_score), 6),
        "average_confidence": round(float(df[confidence_column].mean()), 6),
        "news_count": news_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_ratio": round(positive_count / news_count, 6) if news_count else 0.0,
        "negative_ratio": round(negative_count / news_count, 6) if news_count else 0.0,
        "neutral_ratio": round(neutral_count / news_count, 6) if news_count else 1.0,
        "dominant_label": dominant_label,
    }


def analyze_text_input(text: Any) -> Dict[str, Any]:
    text = clean_text(text)

    if not text:
        return {
            "items": [],
            "table": pd.DataFrame(),
            "summary": aggregate_sentiment(pd.DataFrame()),
        }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        lines = [text]

    sentiment_df = analyze_news_batch(lines)
    summary = aggregate_sentiment(sentiment_df)

    return {
        "items": lines,
        "table": sentiment_df,
        "summary": summary,
    }