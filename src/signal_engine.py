from typing import Any, Dict, Optional

import pandas as pd

from src.config import (
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    STRONG_BUY_THRESHOLD,
    STRONG_SELL_THRESHOLD,
    SENTIMENT_WEIGHT,
    TREND_WEIGHT,
    MOMENTUM_WEIGHT,
    VOLUME_WEIGHT,
    ADVANCED_INDICATOR_WEIGHT,
    RISK_WEIGHT,
    ML_SIGNAL_WEIGHT,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)
from src.indicators import add_all_indicators, get_latest_indicators, summarize_indicator_state
from src.sentiment import aggregate_sentiment
from src.utils import clamp, confidence_from_score, format_pct, safe_float

try:
    from src.indicator_signal_model import generate_advanced_indicator_signal
except Exception:
    generate_advanced_indicator_signal = None

try:
    from src.risk_filters import apply_risk_filters, apply_decision_override
except Exception:
    apply_risk_filters = None
    apply_decision_override = None

try:
    from src.ml_signal import predict_ml_signal
except Exception:
    predict_ml_signal = None


def calculate_sentiment_score(sentiment_summary: Optional[Dict[str, Any]]) -> float:
    if not sentiment_summary:
        return 0.0

    score = safe_float(sentiment_summary.get("sentiment_score"), 0.0)
    news_count = safe_float(sentiment_summary.get("news_count"), 0.0)

    if news_count <= 0:
        return 0.0

    if news_count == 1:
        score *= 0.85

    return clamp(score, -1.0, 1.0)


def calculate_trend_score(indicators: Dict[str, Any]) -> float:
    if not indicators or not indicators.get("available"):
        return 0.0

    score = 0.0

    close = safe_float(indicators.get("close"), 0.0)
    sma20 = safe_float(indicators.get("sma20"), 0.0)
    sma50 = safe_float(indicators.get("sma50"), 0.0)
    sma200 = safe_float(indicators.get("sma200"), 0.0)

    close_above_sma20 = int(safe_float(indicators.get("close_above_sma20"), 0))
    close_above_sma50 = int(safe_float(indicators.get("close_above_sma50"), 0))
    close_above_sma200 = int(safe_float(indicators.get("close_above_sma200"), 0))

    if close_above_sma20:
        score += 0.25
    else:
        score -= 0.25

    if close_above_sma50:
        score += 0.30
    else:
        score -= 0.30

    if sma200 > 0:
        if close_above_sma200:
            score += 0.20
        else:
            score -= 0.20

    if close > sma20 > sma50:
        score += 0.25
    elif close < sma20 < sma50:
        score -= 0.25

    return clamp(score, -1.0, 1.0)


def calculate_momentum_score(indicators: Dict[str, Any]) -> float:
    if not indicators or not indicators.get("available"):
        return 0.0

    score = 0.0

    rsi = safe_float(indicators.get("rsi"), 50.0)
    macd_hist = safe_float(indicators.get("macd_hist"), 0.0)
    macd_bullish = int(safe_float(indicators.get("macd_bullish"), 0))
    distance_sma20 = safe_float(indicators.get("distance_sma20"), 0.0)

    if RSI_OVERSOLD < rsi < RSI_OVERBOUGHT:
        if rsi >= 55:
            score += 0.25
        elif rsi <= 45:
            score -= 0.15
        else:
            score += 0.05

    elif rsi >= RSI_OVERBOUGHT:
        score -= 0.25

        if distance_sma20 > 0.08:
            score -= 0.20

    elif rsi <= RSI_OVERSOLD:
        score += 0.20

    if macd_bullish:
        score += 0.30
    else:
        score -= 0.20

    if macd_hist > 0:
        score += 0.15
    elif macd_hist < 0:
        score -= 0.15

    return clamp(score, -1.0, 1.0)


def calculate_volume_score(indicators: Dict[str, Any]) -> float:
    if not indicators or not indicators.get("available"):
        return 0.0

    volume_spike = safe_float(indicators.get("volume_spike"), 1.0)
    volume_zscore = safe_float(indicators.get("volume_zscore"), 0.0)

    score = 0.0

    if volume_spike >= 2.0:
        score += 0.60
    elif volume_spike >= 1.5:
        score += 0.40
    elif volume_spike >= 1.2:
        score += 0.20
    elif volume_spike <= 0.60:
        score -= 0.25

    if volume_zscore >= 2.0:
        score += 0.20
    elif volume_zscore <= -1.5:
        score -= 0.15

    return clamp(score, -1.0, 1.0)


def calculate_risk_score(
    indicators: Dict[str, Any],
    price_df: Optional[pd.DataFrame] = None,
) -> float:
    if not indicators or not indicators.get("available"):
        return 0.50

    risk = 0.0

    rsi = safe_float(indicators.get("rsi"), 50.0)
    volatility = safe_float(indicators.get("volatility"), 0.0)
    annualized_volatility = safe_float(indicators.get("annualized_volatility"), 0.0)
    atr_pct = safe_float(indicators.get("atr_pct"), 0.0)
    gap = abs(safe_float(indicators.get("gap"), 0.0))
    volume_spike = safe_float(indicators.get("volume_spike"), 1.0)

    if rsi >= 80 or rsi <= 20:
        risk += 0.20
    elif rsi >= 70 or rsi <= 30:
        risk += 0.10

    if annualized_volatility >= 0.70:
        risk += 0.30
    elif annualized_volatility >= 0.45:
        risk += 0.18
    elif volatility >= 0.06:
        risk += 0.12

    if atr_pct >= 0.08:
        risk += 0.25
    elif atr_pct >= 0.045:
        risk += 0.12

    if gap >= 0.08:
        risk += 0.25
    elif gap >= 0.05:
        risk += 0.12

    if volume_spike >= 4.0:
        risk += 0.20
    elif volume_spike <= 0.50:
        risk += 0.10

    return clamp(risk, 0.0, 1.0)


def calculate_final_score(features: Dict[str, Any]) -> float:
    sentiment_score = safe_float(features.get("sentiment_score"), 0.0)
    trend_score = safe_float(features.get("trend_score"), 0.0)
    momentum_score = safe_float(features.get("momentum_score"), 0.0)
    volume_score = safe_float(features.get("volume_score"), 0.0)
    advanced_indicator_score = safe_float(features.get("advanced_indicator_score"), 0.0)
    risk_score = safe_float(features.get("risk_score"), 0.0)
    ml_score = safe_float(features.get("ml_score"), 0.0)

    final_score = (
        SENTIMENT_WEIGHT * sentiment_score
        + TREND_WEIGHT * trend_score
        + MOMENTUM_WEIGHT * momentum_score
        + VOLUME_WEIGHT * volume_score
        + ADVANCED_INDICATOR_WEIGHT * advanced_indicator_score
        - RISK_WEIGHT * risk_score
    )

    if ml_score != 0.0:
        final_score = final_score + (ML_SIGNAL_WEIGHT * ml_score)

    return round(float(clamp(final_score, -1.0, 1.0)), 6)


def label_from_score(score: float) -> str:
    score = safe_float(score, 0.0)

    if score >= STRONG_BUY_THRESHOLD:
        return "STRONG BUY"

    if score >= BUY_THRESHOLD:
        return "BUY"

    if score <= STRONG_SELL_THRESHOLD:
        return "STRONG SELL"

    if score <= SELL_THRESHOLD:
        return "SELL"

    return "HOLD"


def calculate_confidence(
    final_score: float,
    sentiment_summary: Optional[Dict[str, Any]],
    trend_score: float,
    momentum_score: float,
    volume_score: float,
    advanced_indicator_score: float,
    risk_score: float,
    risk_penalty: float = 0.0,
) -> float:
    sentiment_confidence = 0.0
    news_depth_score = 0.0

    if sentiment_summary:
        sentiment_confidence = safe_float(
            sentiment_summary.get(
                "average_confidence",
                sentiment_summary.get("confidence", 0.0),
            ),
            0.0,
        )
        news_count = safe_float(sentiment_summary.get("news_count"), 0.0)
        news_depth_score = min(news_count / 5.0, 1.0)

    indicator_agreement = (
        abs(trend_score)
        + abs(momentum_score)
        + abs(volume_score)
        + abs(advanced_indicator_score)
    ) / 4.0

    risk_safety = 1.0 - clamp(risk_score + risk_penalty, 0.0, 1.0)

    confidence = (
        0.30 * sentiment_confidence
        + 0.25 * abs(final_score)
        + 0.25 * indicator_agreement
        + 0.10 * news_depth_score
        + 0.10 * risk_safety
    )

    return round(float(clamp(confidence, 0.0, 1.0)), 6)


def confidence_label(confidence: float) -> str:
    confidence = safe_float(confidence, 0.0)

    if confidence >= 0.70:
        return "High"

    if confidence >= 0.45:
        return "Medium"

    return "Low"


def generate_signal(
    price_df: pd.DataFrame,
    sentiment_summary: Optional[Dict[str, Any]] = None,
    sentiment_df: Optional[pd.DataFrame] = None,
    ticker: str = "",
    use_ml_model: bool = True,
) -> Dict[str, Any]:
    if price_df is None or price_df.empty:
        return {
            "success": False,
            "signal": "HOLD",
            "final_signal": "HOLD",
            "final_score": 0.0,
            "confidence": 0.0,
            "confidence_label": "Low",
            "message": "Price data is unavailable.",
        }

    enriched_df = add_all_indicators(price_df)

    if enriched_df.empty:
        return {
            "success": False,
            "signal": "HOLD",
            "final_signal": "HOLD",
            "final_score": 0.0,
            "confidence": 0.0,
            "confidence_label": "Low",
            "message": "Could not calculate indicators.",
        }

    if sentiment_summary is None and sentiment_df is not None:
        sentiment_summary = aggregate_sentiment(sentiment_df)

    if sentiment_summary is None:
        sentiment_summary = {
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

    indicators = get_latest_indicators(enriched_df)

    sentiment_score = calculate_sentiment_score(sentiment_summary)
    trend_score = calculate_trend_score(indicators)
    momentum_score = calculate_momentum_score(indicators)
    volume_score = calculate_volume_score(indicators)
    risk_score = calculate_risk_score(indicators, enriched_df)

    indicators["trend_score"] = trend_score
    indicators["momentum_score"] = momentum_score
    indicators["volume_score"] = volume_score
    indicators["risk_score"] = risk_score

    advanced_signal = {
        "available": False,
        "advanced_indicator_score": 0.0,
        "advanced_signal_label": "neutral",
        "advanced_reasons": [],
        "advanced_warnings": [],
        "advanced_indicators": {},
    }

    if generate_advanced_indicator_signal is not None:
        try:
            advanced_signal = generate_advanced_indicator_signal(enriched_df)
        except Exception as error:
            advanced_signal = {
                "available": False,
                "advanced_indicator_score": 0.0,
                "advanced_signal_label": "neutral",
                "advanced_reasons": [],
                "advanced_warnings": [str(error)],
                "advanced_indicators": {},
            }

    advanced_indicator_score = safe_float(
        advanced_signal.get("advanced_indicator_score"),
        0.0,
    )

    ml_result = {
        "available": False,
        "ml_signal": "HOLD",
        "ml_score": 0.0,
        "ml_confidence": 0.0,
        "ml_probabilities": {},
        "message": "ML model not used.",
    }

    ml_score = 0.0

    if use_ml_model and predict_ml_signal is not None:
        try:
            ml_result = predict_ml_signal(
                ticker=ticker,
                price_df=enriched_df,
                sentiment_summary=sentiment_summary,
                indicators=indicators,
                advanced_signal=advanced_signal,
            )
            if ml_result.get("available"):
                ml_score = safe_float(ml_result.get("ml_score"), 0.0)
        except Exception as error:
            ml_result = {
                "available": False,
                "ml_signal": "HOLD",
                "ml_score": 0.0,
                "ml_confidence": 0.0,
                "ml_probabilities": {},
                "message": "ML signal failed.",
                "error": str(error),
            }

    features = {
        "sentiment_score": sentiment_score,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "advanced_indicator_score": advanced_indicator_score,
        "risk_score": risk_score,
        "ml_score": ml_score,
    }

    preliminary_score = calculate_final_score(features)

    risk_result = {
        "risk_flags": [],
        "risk_penalty": 0.0,
        "risk_level": "Low",
        "has_blocker": False,
        "risk_score": risk_score,
    }

    if apply_risk_filters is not None:
        try:
            risk_result = apply_risk_filters(
                sentiment_summary=sentiment_summary,
                indicators=indicators,
                advanced_signal=advanced_signal,
                ml_signal=ml_result,
                final_score=preliminary_score,
            )
        except Exception as error:
            risk_result = {
                "risk_flags": [
                    {
                        "name": "Risk filter error",
                        "severity": "medium",
                        "message": str(error),
                        "blocker": False,
                    }
                ],
                "risk_penalty": 0.10,
                "risk_level": "Medium",
                "has_blocker": False,
                "risk_score": risk_score,
            }

    risk_penalty = safe_float(risk_result.get("risk_penalty"), 0.0)

    final_score = clamp(preliminary_score - risk_penalty, -1.0, 1.0)
    final_score = round(float(final_score), 6)

    raw_signal = label_from_score(final_score)

    confidence = calculate_confidence(
        final_score=final_score,
        sentiment_summary=sentiment_summary,
        trend_score=trend_score,
        momentum_score=momentum_score,
        volume_score=volume_score,
        advanced_indicator_score=advanced_indicator_score,
        risk_score=risk_score,
        risk_penalty=risk_penalty,
    )

    final_signal = raw_signal
    override_result = {
        "original_decision": raw_signal,
        "adjusted_decision": raw_signal,
        "override_applied": False,
        "override_reason": "No override applied.",
    }

    if apply_decision_override is not None:
        override_result = apply_decision_override(
            decision=raw_signal,
            final_score=final_score,
            risk_result=risk_result,
            confidence=confidence,
        )
        final_signal = override_result.get("adjusted_decision", raw_signal)

    summary = summarize_indicator_state(enriched_df)

    return {
        "success": True,
        "ticker": ticker,
        "signal": raw_signal,
        "final_signal": final_signal,
        "final_score": final_score,
        "preliminary_score": preliminary_score,
        "confidence": confidence,
        "confidence_label": confidence_label(confidence),
        "sentiment_score": sentiment_score,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "advanced_indicator_score": advanced_indicator_score,
        "risk_score": risk_score,
        "risk_penalty": risk_penalty,
        "risk_level": risk_result.get("risk_level", "Low"),
        "features": features,
        "sentiment_summary": sentiment_summary,
        "indicators": indicators,
        "indicator_summary": summary,
        "advanced_signal": advanced_signal,
        "ml_signal": ml_result,
        "risk_result": risk_result,
        "override": override_result,
        "enriched_price_df": enriched_df,
        "message": build_signal_explanation(
            final_signal=final_signal,
            raw_signal=raw_signal,
            final_score=final_score,
            sentiment_score=sentiment_score,
            trend_score=trend_score,
            momentum_score=momentum_score,
            advanced_indicator_score=advanced_indicator_score,
            risk_level=risk_result.get("risk_level", "Low"),
            override_result=override_result,
        ),
    }


def build_signal_explanation(
    final_signal: str,
    raw_signal: str,
    final_score: float,
    sentiment_score: float,
    trend_score: float,
    momentum_score: float,
    advanced_indicator_score: float,
    risk_level: str,
    override_result: Dict[str, Any],
) -> str:
    parts = []

    parts.append(f"Final signal is {final_signal} with score {final_score:.3f}.")

    if final_signal != raw_signal:
        parts.append(f"Raw signal was {raw_signal}, but risk override changed it.")

    if sentiment_score > 0.20:
        parts.append("FinBERT sentiment is positive.")
    elif sentiment_score < -0.20:
        parts.append("FinBERT sentiment is negative.")
    else:
        parts.append("FinBERT sentiment is close to neutral.")

    if trend_score > 0.20:
        parts.append("Trend structure is bullish.")
    elif trend_score < -0.20:
        parts.append("Trend structure is bearish.")
    else:
        parts.append("Trend structure is mixed.")

    if momentum_score > 0.20:
        parts.append("Momentum supports upside.")
    elif momentum_score < -0.20:
        parts.append("Momentum is weak.")
    else:
        parts.append("Momentum is neutral.")

    if advanced_indicator_score > 0.20:
        parts.append("Advanced indicators support the signal.")
    elif advanced_indicator_score < -0.20:
        parts.append("Advanced indicators show caution.")
    else:
        parts.append("Advanced indicators are mostly neutral.")

    parts.append(f"Risk level is {risk_level}.")

    if override_result.get("override_applied"):
        parts.append(override_result.get("override_reason", ""))

    return " ".join([p for p in parts if p])


def build_signal_table(signal_result: Dict[str, Any]) -> pd.DataFrame:
    if not signal_result:
        return pd.DataFrame(columns=["Component", "Value"])

    rows = [
        {"Component": "Final Signal", "Value": signal_result.get("final_signal", "HOLD")},
        {"Component": "Raw Signal", "Value": signal_result.get("signal", "HOLD")},
        {"Component": "Final Score", "Value": signal_result.get("final_score", 0.0)},
        {"Component": "Confidence", "Value": signal_result.get("confidence", 0.0)},
        {"Component": "Confidence Label", "Value": signal_result.get("confidence_label", "Low")},
        {"Component": "Sentiment Score", "Value": signal_result.get("sentiment_score", 0.0)},
        {"Component": "Trend Score", "Value": signal_result.get("trend_score", 0.0)},
        {"Component": "Momentum Score", "Value": signal_result.get("momentum_score", 0.0)},
        {"Component": "Volume Score", "Value": signal_result.get("volume_score", 0.0)},
        {"Component": "Advanced Indicator Score", "Value": signal_result.get("advanced_indicator_score", 0.0)},
        {"Component": "Risk Score", "Value": signal_result.get("risk_score", 0.0)},
        {"Component": "Risk Penalty", "Value": signal_result.get("risk_penalty", 0.0)},
        {"Component": "Risk Level", "Value": signal_result.get("risk_level", "Low")},
    ]

    ml_signal = signal_result.get("ml_signal", {})

    if ml_signal and ml_signal.get("available"):
        rows.extend(
            [
                {"Component": "ML Signal", "Value": ml_signal.get("ml_signal", "HOLD")},
                {"Component": "ML Score", "Value": ml_signal.get("ml_score", 0.0)},
                {"Component": "ML Confidence", "Value": ml_signal.get("ml_confidence", 0.0)},
            ]
        )

    return pd.DataFrame(rows)


def build_feature_contribution_table(signal_result: Dict[str, Any]) -> pd.DataFrame:
    if not signal_result:
        return pd.DataFrame(columns=["Feature", "Score", "Weight", "Contribution"])

    rows = []

    mapping = [
        ("Sentiment", "sentiment_score", SENTIMENT_WEIGHT),
        ("Trend", "trend_score", TREND_WEIGHT),
        ("Momentum", "momentum_score", MOMENTUM_WEIGHT),
        ("Volume", "volume_score", VOLUME_WEIGHT),
        ("Advanced Indicators", "advanced_indicator_score", ADVANCED_INDICATOR_WEIGHT),
        ("Risk", "risk_score", -RISK_WEIGHT),
    ]

    for name, key, weight in mapping:
        score = safe_float(signal_result.get(key), 0.0)
        rows.append(
            {
                "Feature": name,
                "Score": round(score, 4),
                "Weight": weight,
                "Contribution": round(score * weight, 4),
            }
        )

    ml_signal = signal_result.get("ml_signal", {})

    if ml_signal and ml_signal.get("available"):
        ml_score = safe_float(ml_signal.get("ml_score"), 0.0)
        rows.append(
            {
                "Feature": "ML Signal",
                "Score": round(ml_score, 4),
                "Weight": ML_SIGNAL_WEIGHT,
                "Contribution": round(ml_score * ML_SIGNAL_WEIGHT, 4),
            }
        )

    return pd.DataFrame(rows)