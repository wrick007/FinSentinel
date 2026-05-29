from typing import Any, Dict, List

from src.advanced_indicators import get_latest_advanced_indicators
from src.utils import clamp, safe_float


def _label_from_score(score: float) -> str:
    score = safe_float(score, 0.0)

    if score >= 0.35:
        return "bullish"

    if score <= -0.35:
        return "bearish"

    return "neutral"


def _add_reason(reasons: List[str], text: str) -> None:
    if text and text not in reasons:
        reasons.append(text)


def _add_warning(warnings: List[str], text: str) -> None:
    if text and text not in warnings:
        warnings.append(text)


def score_fibonacci_zone(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    fib_zone = str(advanced.get("fib_zone", "neutral"))
    nearest_fib_name = str(advanced.get("nearest_fib_name", ""))
    distance = safe_float(advanced.get("nearest_fib_distance"), 1.0)

    if distance <= 0.015:
        if fib_zone == "support":
            score += 0.20
            _add_reason(
                reasons,
                f"Price is very close to Fibonacci {nearest_fib_name}% support.",
            )
        elif fib_zone == "resistance":
            score -= 0.20
            _add_warning(
                warnings,
                f"Price is very close to Fibonacci {nearest_fib_name}% resistance.",
            )

    elif distance <= 0.025:
        if fib_zone == "support":
            score += 0.12
            _add_reason(
                reasons,
                f"Price is near Fibonacci {nearest_fib_name}% support.",
            )
        elif fib_zone == "resistance":
            score -= 0.12
            _add_warning(
                warnings,
                f"Price is near Fibonacci {nearest_fib_name}% resistance.",
            )

    return score, reasons, warnings


def score_adx_trend(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    adx = safe_float(advanced.get("adx"), 0.0)
    adx_bullish = int(safe_float(advanced.get("adx_bullish"), 0))
    plus_di = safe_float(advanced.get("plus_di"), 0.0)
    minus_di = safe_float(advanced.get("minus_di"), 0.0)

    if adx >= 30:
        if adx_bullish:
            score += 0.22
            _add_reason(
                reasons,
                "ADX shows a strong bullish trend.",
            )
        else:
            score -= 0.22
            _add_warning(
                warnings,
                "ADX shows a strong bearish trend.",
            )

    elif adx >= 20:
        if adx_bullish:
            score += 0.12
            _add_reason(
                reasons,
                "ADX shows a developing bullish trend.",
            )
        else:
            score -= 0.12
            _add_warning(
                warnings,
                "ADX shows a developing bearish trend.",
            )

    else:
        _add_warning(
            warnings,
            "ADX trend strength is weak, so directional conviction is limited.",
        )

    if plus_di > minus_di:
        score += 0.04
    elif minus_di > plus_di:
        score -= 0.04

    return score, reasons, warnings


def score_oscillators(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    stoch_k = safe_float(advanced.get("stoch_k"), 50.0)
    stoch_d = safe_float(advanced.get("stoch_d"), 50.0)
    williams_r = safe_float(advanced.get("williams_r"), -50.0)
    cci = safe_float(advanced.get("cci"), 0.0)

    if stoch_k > stoch_d and 20 <= stoch_k <= 80:
        score += 0.08
        _add_reason(
            reasons,
            "Stochastic oscillator supports improving momentum.",
        )

    elif stoch_k < stoch_d and 20 <= stoch_k <= 80:
        score -= 0.08
        _add_warning(
            warnings,
            "Stochastic oscillator shows weakening momentum.",
        )

    if stoch_k >= 85:
        score -= 0.10
        _add_warning(
            warnings,
            "Stochastic oscillator is overbought.",
        )

    elif stoch_k <= 15:
        score += 0.06
        _add_reason(
            reasons,
            "Stochastic oscillator is oversold, suggesting possible rebound.",
        )

    if williams_r >= -20:
        score -= 0.08
        _add_warning(
            warnings,
            "Williams %R is in overbought territory.",
        )

    elif williams_r <= -80:
        score += 0.06
        _add_reason(
            reasons,
            "Williams %R is in oversold territory.",
        )

    if cci >= 100:
        score += 0.10
        _add_reason(
            reasons,
            "CCI confirms strong bullish momentum.",
        )

    elif cci <= -100:
        score -= 0.10
        _add_warning(
            warnings,
            "CCI confirms strong bearish momentum.",
        )

    return score, reasons, warnings


def score_volume_structure(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    obv_trend = int(safe_float(advanced.get("obv_trend"), 0))
    close_above_vwap = int(safe_float(advanced.get("close_above_vwap"), 0))
    close = safe_float(advanced.get("close"), 0.0)
    vwap = safe_float(advanced.get("vwap_approx"), 0.0)

    if obv_trend > 0:
        score += 0.08
        _add_reason(
            reasons,
            "OBV trend supports accumulation.",
        )

    elif obv_trend < 0:
        score -= 0.08
        _add_warning(
            warnings,
            "OBV trend suggests distribution.",
        )

    if close_above_vwap:
        score += 0.08
        _add_reason(
            reasons,
            "Price is above approximate VWAP.",
        )

    else:
        if close > 0 and vwap > 0:
            score -= 0.08
            _add_warning(
                warnings,
                "Price is below approximate VWAP.",
            )

    return score, reasons, warnings


def score_donchian_support_resistance(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    breakout_up = int(safe_float(advanced.get("donchian_breakout_up"), 0))
    breakout_down = int(safe_float(advanced.get("donchian_breakout_down"), 0))
    donchian_position = safe_float(advanced.get("donchian_position"), 0.5)
    near_support = int(safe_float(advanced.get("near_support"), 0))
    near_resistance = int(safe_float(advanced.get("near_resistance"), 0))

    if breakout_up:
        score += 0.18
        _add_reason(
            reasons,
            "Donchian channel shows upside breakout.",
        )

    if breakout_down:
        score -= 0.18
        _add_warning(
            warnings,
            "Donchian channel shows downside breakdown.",
        )

    if donchian_position >= 0.75:
        score += 0.06
        _add_reason(
            reasons,
            "Price is trading near the upper part of the Donchian channel.",
        )

    elif donchian_position <= 0.25:
        score -= 0.06
        _add_warning(
            warnings,
            "Price is trading near the lower part of the Donchian channel.",
        )

    if near_support:
        score += 0.10
        _add_reason(
            reasons,
            "Price is near rolling support.",
        )

    if near_resistance:
        score -= 0.10
        _add_warning(
            warnings,
            "Price is near rolling resistance.",
        )

    return score, reasons, warnings


def score_volatility_context(advanced: Dict[str, Any]) -> tuple[float, List[str], List[str]]:
    score = 0.0
    reasons = []
    warnings = []

    bb_squeeze = int(safe_float(advanced.get("bb_squeeze"), 0))
    volatility_regime = str(advanced.get("volatility_regime", "normal"))
    volatility_z = safe_float(advanced.get("volatility_regime_z"), 0.0)

    if bb_squeeze:
        score += 0.04
        _add_reason(
            reasons,
            "Bollinger squeeze detected, indicating possible breakout setup.",
        )

    if volatility_regime == "high":
        score -= 0.10
        _add_warning(
            warnings,
            "Volatility regime is high, reducing signal reliability.",
        )

    elif volatility_regime == "low":
        score += 0.04
        _add_reason(
            reasons,
            "Volatility regime is low, suggesting a more stable setup.",
        )

    if volatility_z >= 2.0:
        score -= 0.08
        _add_warning(
            warnings,
            "Volatility is unusually elevated compared to its recent baseline.",
        )

    return score, reasons, warnings


def calculate_advanced_indicator_score(advanced: Dict[str, Any]) -> Dict[str, Any]:
    if not advanced or not advanced.get("available"):
        return {
            "available": False,
            "advanced_indicator_score": 0.0,
            "advanced_signal_label": "neutral",
            "advanced_reasons": [],
            "advanced_warnings": ["Advanced indicators are unavailable."],
            "components": {},
        }

    components = {}
    all_reasons = []
    all_warnings = []

    scoring_functions = {
        "fibonacci": score_fibonacci_zone,
        "adx_trend": score_adx_trend,
        "oscillators": score_oscillators,
        "volume_structure": score_volume_structure,
        "donchian_support_resistance": score_donchian_support_resistance,
        "volatility_context": score_volatility_context,
    }

    raw_score = 0.0

    for name, func in scoring_functions.items():
        score, reasons, warnings = func(advanced)
        score = clamp(score, -1.0, 1.0)

        components[name] = round(float(score), 4)
        raw_score += score

        all_reasons.extend(reasons)
        all_warnings.extend(warnings)

    final_score = clamp(raw_score, -1.0, 1.0)
    label = _label_from_score(final_score)

    return {
        "available": True,
        "advanced_indicator_score": round(float(final_score), 4),
        "advanced_signal_label": label,
        "advanced_reasons": all_reasons,
        "advanced_warnings": all_warnings,
        "components": components,
    }


def generate_advanced_indicator_signal(price_df) -> Dict[str, Any]:
    advanced = get_latest_advanced_indicators(price_df)
    score_result = calculate_advanced_indicator_score(advanced)

    return {
        "advanced_indicators": advanced,
        **score_result,
    }


def build_advanced_signal_table(result: Dict[str, Any]):
    import pandas as pd

    if not result or not result.get("available"):
        return pd.DataFrame(
            columns=[
                "Component",
                "Score",
            ]
        )

    components = result.get("components", {})

    rows = [
        {
            "Component": key.replace("_", " ").title(),
            "Score": value,
        }
        for key, value in components.items()
    ]

    rows.append(
        {
            "Component": "Final Advanced Indicator Score",
            "Score": result.get("advanced_indicator_score", 0.0),
        }
    )

    return pd.DataFrame(rows)


def build_advanced_reason_table(result: Dict[str, Any]):
    import pandas as pd

    rows = []

    for reason in result.get("advanced_reasons", []):
        rows.append(
            {
                "Type": "Positive",
                "Message": reason,
            }
        )

    for warning in result.get("advanced_warnings", []):
        rows.append(
            {
                "Type": "Warning",
                "Message": warning,
            }
        )

    return pd.DataFrame(rows)