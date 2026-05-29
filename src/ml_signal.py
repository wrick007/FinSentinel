from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import numpy as np
import pandas as pd

from src.config import SIGNAL_MODEL_DIR
from src.utils import safe_float


DEFAULT_MODEL_PATH = SIGNAL_MODEL_DIR / "signal_model.pkl"

LABEL_ID_TO_NAME = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

LABEL_TO_SCORE = {
    "SELL": -1.0,
    "HOLD": 0.0,
    "BUY": 1.0,
    "STRONG SELL": -1.0,
    "STRONG BUY": 1.0,
}


def _empty_ml_signal(message: str = "ML signal model unavailable.") -> Dict[str, Any]:
    return {
        "available": False,
        "ml_signal": "HOLD",
        "ml_score": 0.0,
        "ml_confidence": 0.0,
        "ml_probabilities": {},
        "model_name": "",
        "feature_columns": [],
        "message": message,
        "error": None,
    }


def load_signal_model_package(
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model_path = Path(model_path or DEFAULT_MODEL_PATH)

    if not model_path.exists():
        return {
            "available": False,
            "error": f"Signal model not found at {model_path}",
            "package": None,
        }

    try:
        with open(model_path, "rb") as file:
            package = pickle.load(file)

        if isinstance(package, dict) and "pipeline" in package:
            return {
                "available": True,
                "error": None,
                "package": package,
            }

        return {
            "available": True,
            "error": None,
            "package": {
                "pipeline": package,
                "feature_columns": [],
                "numeric_features": [],
                "categorical_features": [],
                "model_name": model_path.stem,
                "inverse_label_map": LABEL_ID_TO_NAME,
            },
        }

    except Exception as error:
        return {
            "available": False,
            "error": str(error),
            "package": None,
        }


def _resolve_feature_columns(
    package: Dict[str, Any],
    feature_row: Dict[str, Any],
) -> List[str]:
    feature_columns = package.get("feature_columns") or []

    if feature_columns:
        return [col for col in feature_columns if col in feature_row]

    numeric_features = package.get("numeric_features") or []
    categorical_features = package.get("categorical_features") or []

    inferred = []

    for col in numeric_features + categorical_features:
        if col in feature_row:
            inferred.append(col)

    if inferred:
        return inferred

    excluded = {
        "label",
        "label_id",
        "binary_label",
        "target_return",
        "target_direction",
        "risk_adjusted_future_return",
        "target_horizon",
        "buy_threshold",
        "sell_threshold",
    }

    return [
        key
        for key in feature_row.keys()
        if key not in excluded and not str(key).lower().startswith("future_")
    ]


def _prediction_to_label(
    prediction: Any,
    inverse_label_map: Optional[Dict[Any, Any]] = None,
) -> str:
    inverse_label_map = inverse_label_map or LABEL_ID_TO_NAME

    try:
        if hasattr(prediction, "item"):
            prediction = prediction.item()
    except Exception:
        pass

    if isinstance(prediction, str):
        return prediction.upper()

    try:
        key = int(prediction)
        return str(inverse_label_map.get(key, LABEL_ID_TO_NAME.get(key, "HOLD"))).upper()
    except Exception:
        return "HOLD"


def _probabilities_to_dict(
    probabilities: Optional[np.ndarray],
    classes: Optional[np.ndarray] = None,
    inverse_label_map: Optional[Dict[Any, Any]] = None,
) -> Dict[str, float]:
    if probabilities is None:
        return {}

    inverse_label_map = inverse_label_map or LABEL_ID_TO_NAME

    try:
        probs = np.asarray(probabilities).reshape(-1)
    except Exception:
        return {}

    result = {}

    if classes is not None:
        for cls, prob in zip(classes, probs):
            label = _prediction_to_label(cls, inverse_label_map)
            result[label] = round(float(prob), 6)
    else:
        for idx, prob in enumerate(probs):
            label = _prediction_to_label(idx, inverse_label_map)
            result[label] = round(float(prob), 6)

    return result


def _ml_score_from_probabilities(
    probabilities: Dict[str, float],
    fallback_label: str,
) -> float:
    if probabilities:
        buy_prob = safe_float(probabilities.get("BUY"), 0.0)
        sell_prob = safe_float(probabilities.get("SELL"), 0.0)
        return round(float(buy_prob - sell_prob), 6)

    return LABEL_TO_SCORE.get(str(fallback_label).upper(), 0.0)


def build_live_feature_row(
    ticker: str,
    price_df: pd.DataFrame,
    sentiment_summary: Dict[str, Any],
    indicators: Dict[str, Any],
    advanced_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    advanced_signal = advanced_signal or {}
    advanced_indicators = advanced_signal.get("advanced_indicators", {})

    latest = {}

    if price_df is not None and not price_df.empty:
        row = price_df.iloc[-1]

        for col in price_df.columns:
            value = row.get(col)
            if pd.api.types.is_numeric_dtype(price_df[col]):
                latest[col] = safe_float(value, 0.0)

    feature_row = {
        "ticker": str(ticker).upper(),
        **latest,

        "sentiment_score": safe_float(sentiment_summary.get("sentiment_score"), 0.0),
        "average_confidence": safe_float(sentiment_summary.get("average_confidence"), 0.0),
        "news_count": safe_float(sentiment_summary.get("news_count"), 0.0),
        "positive_count": safe_float(sentiment_summary.get("positive_count"), 0.0),
        "negative_count": safe_float(sentiment_summary.get("negative_count"), 0.0),
        "neutral_count": safe_float(sentiment_summary.get("neutral_count"), 0.0),
        "positive_ratio": safe_float(sentiment_summary.get("positive_ratio"), 0.0),
        "negative_ratio": safe_float(sentiment_summary.get("negative_ratio"), 0.0),
        "neutral_ratio": safe_float(sentiment_summary.get("neutral_ratio"), 0.0),
        "dominant_label": str(sentiment_summary.get("dominant_label", "neutral")).lower(),

        "trend_score": safe_float(indicators.get("trend_score"), 0.0),
        "momentum_score": safe_float(indicators.get("momentum_score"), 0.0),
        "volume_score": safe_float(indicators.get("volume_score"), 0.0),
        "risk_score": safe_float(indicators.get("risk_score"), 0.0),

        "advanced_indicator_score": safe_float(
            advanced_signal.get("advanced_indicator_score"),
            0.0,
        ),
        "adx": safe_float(advanced_indicators.get("adx"), 0.0),
        "stoch_k": safe_float(advanced_indicators.get("stoch_k"), 50.0),
        "stoch_d": safe_float(advanced_indicators.get("stoch_d"), 50.0),
        "williams_r": safe_float(advanced_indicators.get("williams_r"), -50.0),
        "cci": safe_float(advanced_indicators.get("cci"), 0.0),
        "donchian_position": safe_float(advanced_indicators.get("donchian_position"), 0.5),
        "nearest_fib_distance": safe_float(advanced_indicators.get("nearest_fib_distance"), 1.0),
        "close_above_vwap": safe_float(advanced_indicators.get("close_above_vwap"), 0.0),
        "bb_squeeze": safe_float(advanced_indicators.get("bb_squeeze"), 0.0),
        "volatility_regime_z": safe_float(advanced_indicators.get("volatility_regime_z"), 0.0),
    }

    return feature_row


def predict_ml_signal(
    ticker: str,
    price_df: pd.DataFrame,
    sentiment_summary: Dict[str, Any],
    indicators: Dict[str, Any],
    advanced_signal: Optional[Dict[str, Any]] = None,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    model_state = load_signal_model_package(model_path)

    if not model_state.get("available"):
        result = _empty_ml_signal("Trained signal model not found.")
        result["error"] = model_state.get("error")
        return result

    package = model_state["package"]
    pipeline = package.get("pipeline")

    if pipeline is None:
        return _empty_ml_signal("Model package does not contain a pipeline.")

    feature_row = build_live_feature_row(
        ticker=ticker,
        price_df=price_df,
        sentiment_summary=sentiment_summary,
        indicators=indicators,
        advanced_signal=advanced_signal,
    )

    feature_columns = _resolve_feature_columns(package, feature_row)

    if not feature_columns:
        return _empty_ml_signal("No matching feature columns found for live ML inference.")

    X = pd.DataFrame([{col: feature_row.get(col, np.nan) for col in feature_columns}])

    try:
        prediction = pipeline.predict(X)[0]
        inverse_label_map = package.get("inverse_label_map") or LABEL_ID_TO_NAME
        label = _prediction_to_label(prediction, inverse_label_map)

        probabilities = None

        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(X)[0]
            except Exception:
                probabilities = None

        classes = getattr(pipeline, "classes_", None)
        prob_dict = _probabilities_to_dict(
            probabilities,
            classes=classes,
            inverse_label_map=inverse_label_map,
        )

        confidence = max(prob_dict.values()) if prob_dict else 0.0
        ml_score = _ml_score_from_probabilities(prob_dict, label)

        return {
            "available": True,
            "ml_signal": label,
            "ml_score": round(float(ml_score), 6),
            "ml_confidence": round(float(confidence), 6),
            "ml_probabilities": prob_dict,
            "model_name": str(package.get("model_name", "signal_model")),
            "feature_columns": feature_columns,
            "message": "ML signal generated successfully.",
            "error": None,
        }

    except Exception as error:
        result = _empty_ml_signal("ML signal inference failed.")
        result["error"] = str(error)
        result["feature_columns"] = feature_columns
        return result


def build_ml_signal_table(ml_result: Dict[str, Any]) -> pd.DataFrame:
    if not ml_result or not ml_result.get("available", False):
        return pd.DataFrame(
            [
                {
                    "Metric": "Status",
                    "Value": ml_result.get("message", "ML model unavailable.") if ml_result else "ML model unavailable.",
                },
                {
                    "Metric": "Error",
                    "Value": ml_result.get("error", "") if ml_result else "",
                },
            ]
        )

    rows = [
        {
            "Metric": "ML Signal",
            "Value": ml_result.get("ml_signal", "HOLD"),
        },
        {
            "Metric": "ML Score",
            "Value": ml_result.get("ml_score", 0.0),
        },
        {
            "Metric": "ML Confidence",
            "Value": ml_result.get("ml_confidence", 0.0),
        },
        {
            "Metric": "Model Name",
            "Value": ml_result.get("model_name", ""),
        },
        {
            "Metric": "Feature Count",
            "Value": len(ml_result.get("feature_columns", [])),
        },
    ]

    for label, prob in ml_result.get("ml_probabilities", {}).items():
        rows.append(
            {
                "Metric": f"Probability {label}",
                "Value": prob,
            }
        )

    return pd.DataFrame(rows)