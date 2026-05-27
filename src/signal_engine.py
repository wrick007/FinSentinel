from typing import Any, Dict, List, Optional, Tuple

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
    RISK_WEIGHT,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    MAX_ACCEPTABLE_VOLATILITY,
    MIN_AVG_VOLUME,
    EXTREME_GAP_THRESHOLD,
)
from src.indicators import add_all_indicators, get_latest_indicators, summarize_indicator_state
from src.sentiment import aggregate_sentiment
from src.utils import clamp, confidence_from_score, format_pct, safe_float


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

    if volume_spike >= 2.0:
        return 0.75

    if volume_spike >= 1.5:
        return 0.50

    if volume_spike >= 1.2:
        return 0.25

    if volume_spike <= 0.60:
        return -0.25

    return 0.0


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
    distance_sma20 = abs(safe_float(indicators.get("distance_sma20"), 0.0))

    if rsi >= 75 or rsi <= 25:
        risk += 0.20

    if volatility >= MAX_ACCEPTABLE_VOLATILITY:
        risk += 0.25
    elif annualized_volatility >= 0.60:
        risk += 0.20

    if atr_pct >= 0.05:
        risk += 0.20
    elif atr_pct >= 0.03:
        risk += 0.10

    if gap >= EXTREME_GAP_THRESHOLD:
        risk += 0.15

    if distance_sma20 >= 0.10:
        risk += 0.10

    if price_df is not None and not price_df.empty and "Volume" in price_df.columns:
        avg_volume = safe_float(price_df["Volume"].tail(20).mean(), 0.0)

        if avg_volume < MIN_AVG_VOLUME:
            risk += 0.15

    return clamp(risk, 0.0, 1.0)


def calculate_final_score(features: Dict[str, Any]) -> float:
    sentiment_score = safe_float(features.get("sentiment_score"), 0.0)
    trend_score = safe_float(features.get("trend_score"), 0.0)
    momentum_score = safe_float(features.get("momentum_score"), 0.0)
    volume_score = safe_float(features.get("volume_score"), 0.0)
    risk_score = safe_float(features.get("risk_score"), 0.0)

    final_score = (
        SENTIMENT_WEIGHT * sentiment_score
        + TREND_WEIGHT * trend_score
        + MOMENTUM_WEIGHT * momentum_score
        + VOLUME_WEIGHT * volume_score
        - RISK_WEIGHT * risk_score
    )

    return clamp(final_score, -1.0, 1.0)


def label_from_score(final_score: float) -> str:
    final_score = safe_float(final_score, 0.0)

    if final_score >= STRONG_BUY_THRESHOLD:
        return "STRONG BUY"

    if final_score >= BUY_THRESHOLD:
        return "BUY"

    if final_score <= STRONG_SELL_THRESHOLD:
        return "STRONG SELL"

    if final_score <= SELL_THRESHOLD:
        return "SELL"

    return "HOLD"


def risk_label_from_score(risk_score: float) -> str:
    risk_score = safe_float(risk_score, 0.0)

    if risk_score < 0.35:
        return "Low"

    if risk_score < 0.70:
        return "Medium"

    return "High"


def apply_realistic_filters(
    signal: str,
    confidence: float,
    features: Dict[str, Any],
    indicators: Dict[str, Any],
) -> Tuple[str, float, List[str]]:
    reasons = []

    rsi = safe_float(indicators.get("rsi"), 50.0)
    volume_spike = safe_float(indicators.get("volume_spike"), 1.0)
    gap = abs(safe_float(indicators.get("gap"), 0.0))
    risk_score = safe_float(features.get("risk_score"), 0.0)
    sentiment_score = safe_float(features.get("sentiment_score"), 0.0)

    adjusted_signal = signal
    adjusted_confidence = confidence

    if risk_score >= 0.80 and signal in {"BUY", "STRONG BUY"}:
        adjusted_signal = "HOLD"
        adjusted_confidence = min(adjusted_confidence, 55.0)
        reasons.append("Risk is too high for a fresh buy entry.")

    if signal in {"BUY", "STRONG BUY"} and rsi >= 78:
        adjusted_confidence *= 0.75
        reasons.append("RSI is very high, so the buy confidence is reduced.")

    if signal in {"BUY", "STRONG BUY"} and gap >= EXTREME_GAP_THRESHOLD:
        adjusted_confidence *= 0.80
        reasons.append("Large price gap detected, avoiding aggressive chasing.")

    if sentiment_score > 0.40 and volume_spike < 0.80:
        adjusted_confidence *= 0.85
        reasons.append("Positive sentiment has weak volume confirmation.")

    if signal in {"SELL", "STRONG SELL"} and rsi <= 22:
        adjusted_confidence *= 0.80
        reasons.append("RSI is extremely oversold, so sell confidence is reduced.")

    adjusted_confidence = max(5.0, min(95.0, round(adjusted_confidence, 2)))

    return adjusted_signal, adjusted_confidence, reasons


def generate_explanation(
    signal: str,
    features: Dict[str, Any],
    indicators: Dict[str, Any],
    sentiment_summary: Dict[str, Any],
    filter_reasons: Optional[List[str]] = None,
) -> str:
    filter_reasons = filter_reasons or []

    parts = []

    sentiment_score = safe_float(features.get("sentiment_score"), 0.0)
    trend_score = safe_float(features.get("trend_score"), 0.0)
    momentum_score = safe_float(features.get("momentum_score"), 0.0)
    volume_score = safe_float(features.get("volume_score"), 0.0)
    risk_score = safe_float(features.get("risk_score"), 0.0)

    news_count = int(safe_float(sentiment_summary.get("news_count"), 0))
    dominant_label = sentiment_summary.get("dominant_label", "neutral")

    if news_count > 0:
        parts.append(
            f"News sentiment is {dominant_label} across {news_count} item(s), "
            f"with a sentiment score of {sentiment_score:.2f}."
        )
    else:
        parts.append("No reliable news sentiment was available, so the model relied more on price and technical signals.")

    if trend_score > 0.25:
        parts.append("Trend is supportive because price is trading above key moving averages.")
    elif trend_score < -0.25:
        parts.append("Trend is weak because price is below important moving averages.")
    else:
        parts.append("Trend is mixed, so no strong directional confirmation is present.")

    if momentum_score > 0.25:
        parts.append("Momentum is positive based on RSI and MACD conditions.")
    elif momentum_score < -0.25:
        parts.append("Momentum is weak based on RSI and MACD conditions.")
    else:
        parts.append("Momentum is neutral.")

    if volume_score > 0.25:
        parts.append("Volume is stronger than usual, which improves signal confirmation.")
    elif volume_score < -0.10:
        parts.append("Volume confirmation is weak.")
    else:
        parts.append("Volume is normal.")

    risk_label = risk_label_from_score(risk_score)
    parts.append(f"Risk level is {risk_label}.")

    for reason in filter_reasons:
        parts.append(reason)

    if signal in {"BUY", "STRONG BUY"}:
        parts.append("Final view: bullish setup, but entry should still be managed with risk control.")
    elif signal in {"SELL", "STRONG SELL"}:
        parts.append("Final view: bearish setup or weak conditions, so avoiding or exiting may be safer.")
    else:
        parts.append("Final view: no strong edge, so HOLD is preferred.")

    return " ".join(parts)


def generate_signal(
    sentiment_summary: Optional[Dict[str, Any]],
    price_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if sentiment_summary is None:
        sentiment_summary = aggregate_sentiment(pd.DataFrame())

    if price_df is None or price_df.empty:
        sentiment_score = calculate_sentiment_score(sentiment_summary)

        features = {
            "sentiment_score": sentiment_score,
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volume_score": 0.0,
            "risk_score": 0.50,
        }

        final_score = calculate_final_score(features)
        signal = label_from_score(final_score)
        confidence = confidence_from_score(final_score)

        explanation = (
            "Price data was unavailable, so this signal is based mainly on sentiment. "
            "Use caution because technical confirmation could not be checked."
        )

        return {
            "signal": signal,
            "confidence": confidence,
            "risk_label": risk_label_from_score(0.50),
            "final_score": round(final_score, 6),
            "features": features,
            "indicators": {},
            "indicator_state": {},
            "explanation": explanation,
            "success": True,
        }

    price_df = add_all_indicators(price_df)
    indicators = get_latest_indicators(price_df)
    indicator_state = summarize_indicator_state(price_df)

    sentiment_score = calculate_sentiment_score(sentiment_summary)
    trend_score = calculate_trend_score(indicators)
    momentum_score = calculate_momentum_score(indicators)
    volume_score = calculate_volume_score(indicators)
    risk_score = calculate_risk_score(indicators, price_df)

    features = {
        "sentiment_score": sentiment_score,
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "risk_score": risk_score,
    }

    final_score = calculate_final_score(features)
    raw_signal = label_from_score(final_score)
    raw_confidence = confidence_from_score(final_score)

    signal, confidence, filter_reasons = apply_realistic_filters(
        signal=raw_signal,
        confidence=raw_confidence,
        features=features,
        indicators=indicators,
    )

    explanation = generate_explanation(
        signal=signal,
        features=features,
        indicators=indicators,
        sentiment_summary=sentiment_summary,
        filter_reasons=filter_reasons,
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "risk_label": risk_label_from_score(risk_score),
        "final_score": round(final_score, 6),
        "raw_signal": raw_signal,
        "raw_confidence": raw_confidence,
        "features": features,
        "indicators": indicators,
        "indicator_state": indicator_state,
        "explanation": explanation,
        "success": True,
    }


def generate_signal_from_sentiment_df(
    sentiment_df: Optional[pd.DataFrame],
    price_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    sentiment_summary = aggregate_sentiment(sentiment_df)
    return generate_signal(sentiment_summary=sentiment_summary, price_df=price_df)


def build_signal_table(result: Dict[str, Any]) -> pd.DataFrame:
    features = result.get("features", {}) or {}
    indicators = result.get("indicators", {}) or {}
    indicator_state = result.get("indicator_state", {}) or {}

    rows = [
        {"Metric": "Signal", "Value": result.get("signal", "HOLD")},
        {"Metric": "Confidence", "Value": f"{safe_float(result.get('confidence'), 0.0):.2f}%"},
        {"Metric": "Risk Level", "Value": result.get("risk_label", "Medium")},
        {"Metric": "Final Score", "Value": f"{safe_float(result.get('final_score'), 0.0):.3f}"},
        {"Metric": "Sentiment Score", "Value": f"{safe_float(features.get('sentiment_score'), 0.0):.3f}"},
        {"Metric": "Trend Score", "Value": f"{safe_float(features.get('trend_score'), 0.0):.3f}"},
        {"Metric": "Momentum Score", "Value": f"{safe_float(features.get('momentum_score'), 0.0):.3f}"},
        {"Metric": "Volume Score", "Value": f"{safe_float(features.get('volume_score'), 0.0):.3f}"},
        {"Metric": "Risk Score", "Value": f"{safe_float(features.get('risk_score'), 0.0):.3f}"},
        {"Metric": "RSI", "Value": f"{safe_float(indicators.get('rsi'), 50.0):.2f}"},
        {"Metric": "Volume Spike", "Value": f"{safe_float(indicators.get('volume_spike'), 1.0):.2f}x"},
        {"Metric": "Volatility", "Value": format_pct(safe_float(indicators.get("volatility"), 0.0))},
        {"Metric": "Trend State", "Value": indicator_state.get("trend_state", "Unknown")},
        {"Metric": "Momentum State", "Value": indicator_state.get("momentum_state", "Unknown")},
    ]

    return pd.DataFrame(rows)