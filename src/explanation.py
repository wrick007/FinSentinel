from typing import Any, Dict


def generate_rule_based_explanation(payload: Dict[str, Any]) -> str:
    final_signal = payload.get("final_signal", "HOLD")
    confidence = payload.get("confidence", 0)

    sentiment = payload.get("sentiment", {})
    technical = payload.get("technical", {})
    ml_model = payload.get("ml_model", {})
    risk = payload.get("risk", {})

    reasons = []

    sentiment_label = sentiment.get("label", "neutral")
    sentiment_score = float(sentiment.get("score", 0) or 0)

    if sentiment_label == "positive" or sentiment_score > 0.25:
        reasons.append("news sentiment is positive")
    elif sentiment_label == "negative" or sentiment_score < -0.25:
        reasons.append("news sentiment is negative")
    else:
        reasons.append("news sentiment is mostly neutral")

    rsi = technical.get("rsi")

    if rsi is not None:
        try:
            rsi = float(rsi)

            if rsi > 70:
                reasons.append("RSI suggests overbought conditions")
            elif rsi < 30:
                reasons.append("RSI suggests oversold conditions")
            else:
                reasons.append("RSI is in a normal range")

        except Exception:
            pass

    if technical.get("price_above_sma50") is True:
        reasons.append("price is above the 50-day moving average")
    elif technical.get("price_above_sma50") is False:
        reasons.append("price is below the 50-day moving average")

    ml_signal = ml_model.get("signal")

    if ml_signal:
        reasons.append(f"ML model signal is {ml_signal}")

    risk_level = risk.get("risk_level", "unknown")

    if not reasons:
        reasons.append("available model and indicator values were used")

    return f"""
Signal Summary:
The final model-generated signal is {final_signal} with confidence {confidence}.

Main Reasons:
This signal is based on the fact that {", ".join(reasons)}.

Risk Factors:
The current risk level is {risk_level}. This signal can be wrong if market conditions, volatility, liquidity, or news flow changes suddenly.

Final View:
Use this as an educational model-assisted signal, not guaranteed financial advice.
""".strip()