from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils import clamp, safe_float


def _add_flag(
    flags: List[Dict[str, Any]],
    name: str,
    severity: str,
    message: str,
    blocker: bool = False,
) -> None:
    flags.append(
        {
            "name": name,
            "severity": severity,
            "message": message,
            "blocker": blocker,
        }
    )


def _severity_penalty(severity: str) -> float:
    severity = str(severity).lower().strip()

    if severity == "high":
        return 0.20

    if severity == "medium":
        return 0.10

    if severity == "low":
        return 0.04

    return 0.0


def calculate_risk_penalty(flags: List[Dict[str, Any]]) -> float:
    penalty = 0.0

    for flag in flags:
        penalty += _severity_penalty(flag.get("severity", "low"))

    return round(float(min(penalty, 0.70)), 4)


def has_blocker(flags: List[Dict[str, Any]]) -> bool:
    return any(bool(flag.get("blocker", False)) for flag in flags)


def get_risk_level(risk_penalty: float, flags: Optional[List[Dict[str, Any]]] = None) -> str:
    risk_penalty = safe_float(risk_penalty, 0.0)
    flags = flags or []

    if has_blocker(flags):
        return "High"

    if risk_penalty >= 0.35:
        return "High"

    if risk_penalty >= 0.15:
        return "Medium"

    return "Low"


def apply_sentiment_risk_filters(
    sentiment_summary: Optional[Dict[str, Any]],
    flags: List[Dict[str, Any]],
) -> None:
    if not sentiment_summary:
        _add_flag(
            flags,
            name="Missing sentiment",
            severity="medium",
            message="Sentiment summary is unavailable.",
            blocker=False,
        )
        return

    sentiment_score = safe_float(sentiment_summary.get("sentiment_score"), 0.0)
    average_confidence = safe_float(
        sentiment_summary.get(
            "average_confidence",
            sentiment_summary.get("confidence", 0.0),
        ),
        0.0,
    )
    news_count = int(safe_float(sentiment_summary.get("news_count"), 0))

    if news_count <= 0:
        _add_flag(
            flags,
            name="No news input",
            severity="medium",
            message="No valid news/headlines were used for sentiment.",
            blocker=False,
        )

    elif news_count < 2:
        _add_flag(
            flags,
            name="Low news coverage",
            severity="medium",
            message="Signal is based on very few news items.",
            blocker=False,
        )

    if average_confidence < 0.45:
        _add_flag(
            flags,
            name="Weak FinBERT confidence",
            severity="high",
            message="FinBERT confidence is weak, so sentiment reliability is low.",
            blocker=True,
        )

    elif average_confidence < 0.60:
        _add_flag(
            flags,
            name="Moderate FinBERT confidence",
            severity="medium",
            message="FinBERT confidence is moderate, so the signal should be treated carefully.",
            blocker=False,
        )

    if abs(sentiment_score) < 0.10:
        _add_flag(
            flags,
            name="Neutral sentiment zone",
            severity="low",
            message="Aggregated sentiment is close to neutral.",
            blocker=False,
        )


def apply_indicator_risk_filters(
    indicators: Optional[Dict[str, Any]],
    flags: List[Dict[str, Any]],
) -> None:
    if not indicators or not indicators.get("available", True):
        _add_flag(
            flags,
            name="Missing indicators",
            severity="high",
            message="Technical indicators are unavailable.",
            blocker=True,
        )
        return

    rsi = safe_float(indicators.get("rsi", indicators.get("RSI", 50.0)), 50.0)
    volatility = safe_float(
        indicators.get(
            "annualized_volatility",
            indicators.get("volatility", indicators.get("Volatility_20D", 0.0)),
        ),
        0.0,
    )
    atr_pct = safe_float(indicators.get("atr_pct", indicators.get("ATR_Pct", 0.0)), 0.0)
    gap = abs(safe_float(indicators.get("gap", indicators.get("Gap", 0.0)), 0.0))
    volume_spike = safe_float(
        indicators.get("volume_spike", indicators.get("Volume_Spike", 1.0)),
        1.0,
    )
    avg_volume = safe_float(
        indicators.get("avg_volume", indicators.get("Volume_MA_20", 0.0)),
        0.0,
    )

    if volatility >= 0.75:
        _add_flag(
            flags,
            name="Extreme volatility",
            severity="high",
            message="Annualized volatility is extremely high.",
            blocker=True,
        )

    elif volatility >= 0.45:
        _add_flag(
            flags,
            name="High volatility",
            severity="medium",
            message="Recent volatility is elevated.",
            blocker=False,
        )

    if atr_pct >= 0.08:
        _add_flag(
            flags,
            name="High ATR risk",
            severity="high",
            message="ATR percentage is very high, indicating unstable price movement.",
            blocker=True,
        )

    elif atr_pct >= 0.045:
        _add_flag(
            flags,
            name="Elevated ATR",
            severity="medium",
            message="ATR percentage is elevated.",
            blocker=False,
        )

    if gap >= 0.08:
        _add_flag(
            flags,
            name="Extreme gap risk",
            severity="high",
            message="Large price gap detected.",
            blocker=True,
        )

    elif gap >= 0.05:
        _add_flag(
            flags,
            name="Large gap",
            severity="medium",
            message="Notable price gap detected.",
            blocker=False,
        )

    if avg_volume > 0 and avg_volume < 100000:
        _add_flag(
            flags,
            name="Low liquidity proxy",
            severity="medium",
            message="Average volume is low, which may increase slippage risk.",
            blocker=False,
        )

    if volume_spike <= 0.50:
        _add_flag(
            flags,
            name="Weak volume confirmation",
            severity="medium",
            message="Current volume is weak compared to recent average.",
            blocker=False,
        )

    if volume_spike >= 4.0:
        _add_flag(
            flags,
            name="Abnormal volume shock",
            severity="high",
            message="Very large volume spike may indicate event-driven instability.",
            blocker=True,
        )

    if rsi >= 80:
        _add_flag(
            flags,
            name="Extreme overbought RSI",
            severity="medium",
            message="RSI is extremely overbought.",
            blocker=False,
        )

    elif rsi >= 70:
        _add_flag(
            flags,
            name="Overbought RSI",
            severity="low",
            message="RSI is overbought.",
            blocker=False,
        )

    if rsi <= 20:
        _add_flag(
            flags,
            name="Extreme oversold RSI",
            severity="medium",
            message="RSI is extremely oversold.",
            blocker=False,
        )

    elif rsi <= 30:
        _add_flag(
            flags,
            name="Oversold RSI",
            severity="low",
            message="RSI is oversold.",
            blocker=False,
        )


def apply_advanced_risk_filters(
    advanced_signal: Optional[Dict[str, Any]],
    flags: List[Dict[str, Any]],
) -> None:
    if not advanced_signal:
        return

    advanced_indicators = advanced_signal.get("advanced_indicators", advanced_signal)

    if not advanced_indicators or not advanced_indicators.get("available", False):
        return

    volatility_regime = str(advanced_indicators.get("volatility_regime", "normal"))
    volatility_z = safe_float(advanced_indicators.get("volatility_regime_z"), 0.0)
    bb_squeeze = int(safe_float(advanced_indicators.get("bb_squeeze"), 0))
    near_resistance = int(safe_float(advanced_indicators.get("near_resistance"), 0))
    near_support = int(safe_float(advanced_indicators.get("near_support"), 0))
    fib_zone = str(advanced_indicators.get("fib_zone", "neutral"))
    fib_distance = safe_float(advanced_indicators.get("nearest_fib_distance"), 1.0)
    adx = safe_float(advanced_indicators.get("adx"), 0.0)

    if volatility_regime == "high":
        _add_flag(
            flags,
            name="High volatility regime",
            severity="medium",
            message="Advanced volatility regime is high.",
            blocker=False,
        )

    if volatility_z >= 2.0:
        _add_flag(
            flags,
            name="Volatility expansion",
            severity="high",
            message="Volatility is unusually elevated versus recent baseline.",
            blocker=True,
        )

    if bb_squeeze:
        _add_flag(
            flags,
            name="Bollinger squeeze",
            severity="low",
            message="Bollinger squeeze detected; breakout direction may be uncertain.",
            blocker=False,
        )

    if near_resistance:
        _add_flag(
            flags,
            name="Near resistance",
            severity="medium",
            message="Price is near rolling resistance.",
            blocker=False,
        )

    if near_support:
        _add_flag(
            flags,
            name="Near support",
            severity="low",
            message="Price is near rolling support.",
            blocker=False,
        )

    if fib_zone == "resistance" and fib_distance <= 0.02:
        _add_flag(
            flags,
            name="Fibonacci resistance",
            severity="medium",
            message="Price is very close to a Fibonacci resistance zone.",
            blocker=False,
        )

    if fib_zone == "support" and fib_distance <= 0.02:
        _add_flag(
            flags,
            name="Fibonacci support",
            severity="low",
            message="Price is close to a Fibonacci support zone.",
            blocker=False,
        )

    if adx < 15:
        _add_flag(
            flags,
            name="Weak trend strength",
            severity="low",
            message="ADX suggests weak trend strength.",
            blocker=False,
        )


def apply_signal_conflict_filters(
    sentiment_summary: Optional[Dict[str, Any]],
    indicators: Optional[Dict[str, Any]],
    advanced_signal: Optional[Dict[str, Any]],
    flags: List[Dict[str, Any]],
) -> None:
    sentiment_score = 0.0

    if sentiment_summary:
        sentiment_score = safe_float(sentiment_summary.get("sentiment_score"), 0.0)

    momentum_score = 0.0
    trend_score = 0.0

    if indicators:
        momentum_score = safe_float(indicators.get("momentum_score"), 0.0)
        trend_score = safe_float(indicators.get("trend_score"), 0.0)

        if momentum_score == 0.0:
            rsi = safe_float(indicators.get("rsi", indicators.get("RSI", 50.0)), 50.0)
            macd_bullish = int(safe_float(indicators.get("macd_bullish", 0), 0))

            if rsi >= 55 and macd_bullish:
                momentum_score = 0.5
            elif rsi <= 45 and not macd_bullish:
                momentum_score = -0.5

    advanced_score = 0.0

    if advanced_signal:
        advanced_score = safe_float(advanced_signal.get("advanced_indicator_score"), 0.0)

    if sentiment_score >= 0.35 and trend_score <= -0.35:
        _add_flag(
            flags,
            name="Positive sentiment vs weak trend",
            severity="high",
            message="Sentiment is positive but trend structure is bearish.",
            blocker=True,
        )

    if sentiment_score <= -0.35 and trend_score >= 0.35:
        _add_flag(
            flags,
            name="Negative sentiment vs strong trend",
            severity="medium",
            message="Sentiment is negative but trend structure remains bullish.",
            blocker=False,
        )

    if sentiment_score >= 0.35 and momentum_score <= -0.35:
        _add_flag(
            flags,
            name="Positive sentiment vs bearish momentum",
            severity="high",
            message="Sentiment is positive but momentum is bearish.",
            blocker=True,
        )

    if sentiment_score <= -0.35 and momentum_score >= 0.35:
        _add_flag(
            flags,
            name="Negative sentiment vs bullish momentum",
            severity="medium",
            message="Sentiment is negative but momentum remains bullish.",
            blocker=False,
        )

    if advanced_score <= -0.50 and sentiment_score >= 0.30:
        _add_flag(
            flags,
            name="Advanced indicators disagree with sentiment",
            severity="medium",
            message="Advanced indicator model is bearish while sentiment is positive.",
            blocker=False,
        )

    if advanced_score >= 0.50 and sentiment_score <= -0.30:
        _add_flag(
            flags,
            name="Advanced indicators disagree with negative sentiment",
            severity="medium",
            message="Advanced indicator model is bullish while sentiment is negative.",
            blocker=False,
        )


def apply_ml_signal_risk_filters(
    ml_signal: Optional[Dict[str, Any]],
    final_score: float,
    flags: List[Dict[str, Any]],
) -> None:
    if not ml_signal or not ml_signal.get("available", False):
        return

    ml_label = str(ml_signal.get("ml_signal", "HOLD")).upper()
    ml_confidence = safe_float(ml_signal.get("ml_confidence"), 0.0)
    final_score = safe_float(final_score, 0.0)

    if ml_confidence < 0.45:
        _add_flag(
            flags,
            name="Weak ML signal confidence",
            severity="low",
            message="Trained signal model confidence is low.",
            blocker=False,
        )

    if final_score >= 0.30 and ml_label == "SELL" and ml_confidence >= 0.60:
        _add_flag(
            flags,
            name="ML disagreement",
            severity="medium",
            message="Rule-based signal is bullish but ML model suggests SELL.",
            blocker=False,
        )

    if final_score <= -0.30 and ml_label == "BUY" and ml_confidence >= 0.60:
        _add_flag(
            flags,
            name="ML disagreement",
            severity="medium",
            message="Rule-based signal is bearish but ML model suggests BUY.",
            blocker=False,
        )


def apply_risk_filters(
    sentiment_summary: Optional[Dict[str, Any]] = None,
    indicators: Optional[Dict[str, Any]] = None,
    advanced_signal: Optional[Dict[str, Any]] = None,
    ml_signal: Optional[Dict[str, Any]] = None,
    final_score: float = 0.0,
) -> Dict[str, Any]:
    flags: List[Dict[str, Any]] = []

    apply_sentiment_risk_filters(sentiment_summary, flags)
    apply_indicator_risk_filters(indicators, flags)
    apply_advanced_risk_filters(advanced_signal, flags)
    apply_signal_conflict_filters(sentiment_summary, indicators, advanced_signal, flags)
    apply_ml_signal_risk_filters(ml_signal, final_score, flags)

    penalty = calculate_risk_penalty(flags)
    blocker = has_blocker(flags)
    risk_level = get_risk_level(penalty, flags)

    return {
        "risk_flags": flags,
        "risk_penalty": penalty,
        "risk_level": risk_level,
        "has_blocker": blocker,
        "risk_score": clamp(penalty, 0.0, 1.0),
    }


def apply_decision_override(
    decision: str,
    final_score: float,
    risk_result: Dict[str, Any],
    confidence: float = 0.0,
) -> Dict[str, Any]:
    original_decision = str(decision).upper()
    adjusted_decision = original_decision
    reason = ""

    final_score = safe_float(final_score, 0.0)
    confidence = safe_float(confidence, 0.0)

    if risk_result.get("has_blocker", False):
        adjusted_decision = "HOLD"
        reason = "High-risk blocker triggered, so directional signal is changed to HOLD."

    elif confidence < 0.35:
        adjusted_decision = "HOLD"
        reason = "Confidence is too low for a directional signal."

    elif risk_result.get("risk_level") == "High" and original_decision in {"BUY", "SELL", "STRONG BUY", "STRONG SELL"}:
        adjusted_decision = "HOLD"
        reason = "Overall risk level is high, so directional signal is changed to HOLD."

    elif -0.30 < final_score < 0.30:
        adjusted_decision = "HOLD"
        reason = "Final score is not strong enough for BUY or SELL."

    else:
        reason = "No hard risk override was applied."

    return {
        "original_decision": original_decision,
        "adjusted_decision": adjusted_decision,
        "override_applied": adjusted_decision != original_decision,
        "override_reason": reason,
    }


def build_risk_table(risk_result: Dict[str, Any]) -> pd.DataFrame:
    flags = risk_result.get("risk_flags", [])

    if not flags:
        return pd.DataFrame(
            [
                {
                    "Name": "No major risk",
                    "Severity": "low",
                    "Message": "No major risk filters were triggered.",
                    "Blocker": False,
                }
            ]
        )

    rows = []

    for flag in flags:
        rows.append(
            {
                "Name": flag.get("name", ""),
                "Severity": flag.get("severity", ""),
                "Message": flag.get("message", ""),
                "Blocker": bool(flag.get("blocker", False)),
            }
        )

    return pd.DataFrame(rows)