from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.market_data import clean_price_data
from src.utils import clean_numeric_series, safe_float


FIB_LOOKBACK = 120
SUPPORT_RESISTANCE_WINDOW = 20
ADX_WINDOW = 14
STOCH_WINDOW = 14
CCI_WINDOW = 20
DONCHIAN_WINDOW = 20
VWAP_WINDOW = 20
BOLLINGER_SQUEEZE_WINDOW = 20


def _empty_advanced_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    if series is None or series.empty:
        return default

    value = series.dropna()

    if value.empty:
        return default

    return safe_float(value.iloc[-1], default)


def add_fibonacci_levels(df: pd.DataFrame, lookback: int = FIB_LOOKBACK) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    high_roll = data["High"].rolling(window=lookback, min_periods=max(20, lookback // 4)).max()
    low_roll = data["Low"].rolling(window=lookback, min_periods=max(20, lookback // 4)).min()

    price_range = high_roll - low_roll
    price_range = price_range.replace(0, np.nan)

    data["Fib_High"] = high_roll
    data["Fib_Low"] = low_roll
    data["Fib_236"] = high_roll - 0.236 * price_range
    data["Fib_382"] = high_roll - 0.382 * price_range
    data["Fib_500"] = high_roll - 0.500 * price_range
    data["Fib_618"] = high_roll - 0.618 * price_range
    data["Fib_786"] = high_roll - 0.786 * price_range

    fib_cols = ["Fib_236", "Fib_382", "Fib_500", "Fib_618", "Fib_786"]

    nearest_levels = []
    nearest_distances = []
    nearest_names = []

    for _, row in data.iterrows():
        close = safe_float(row.get("Close"), np.nan)

        if np.isnan(close) or close == 0:
            nearest_levels.append(np.nan)
            nearest_distances.append(np.nan)
            nearest_names.append("")
            continue

        distances = {}

        for col in fib_cols:
            level = safe_float(row.get(col), np.nan)

            if not np.isnan(level):
                distances[col] = abs(close - level) / close

        if not distances:
            nearest_levels.append(np.nan)
            nearest_distances.append(np.nan)
            nearest_names.append("")
            continue

        nearest_col = min(distances, key=distances.get)

        nearest_levels.append(safe_float(row.get(nearest_col), np.nan))
        nearest_distances.append(distances[nearest_col])
        nearest_names.append(nearest_col.replace("Fib_", ""))

    data["Nearest_Fib_Level"] = nearest_levels
    data["Nearest_Fib_Distance"] = nearest_distances
    data["Nearest_Fib_Name"] = nearest_names

    return data


def add_adx(df: pd.DataFrame, window: int = ADX_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    high = clean_numeric_series(data["High"])
    low = clean_numeric_series(data["Low"])
    close = clean_numeric_series(data["Close"])

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_1 = high - low
    tr_2 = (high - close.shift(1)).abs()
    tr_3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)

    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=data.index).ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=data.index).ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    data["ADX"] = adx.replace([np.inf, -np.inf], np.nan)
    data["Plus_DI"] = plus_di.replace([np.inf, -np.inf], np.nan)
    data["Minus_DI"] = minus_di.replace([np.inf, -np.inf], np.nan)
    data["ADX_Bullish"] = (data["Plus_DI"] > data["Minus_DI"]).astype(int)

    return data


def add_stochastic(df: pd.DataFrame, window: int = STOCH_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    low_min = data["Low"].rolling(window=window, min_periods=max(5, window // 2)).min()
    high_max = data["High"].rolling(window=window, min_periods=max(5, window // 2)).max()

    denominator = (high_max - low_min).replace(0, np.nan)

    data["Stoch_K"] = 100 * (data["Close"] - low_min) / denominator
    data["Stoch_D"] = data["Stoch_K"].rolling(window=3, min_periods=2).mean()

    data["Stoch_K"] = data["Stoch_K"].clip(0, 100)
    data["Stoch_D"] = data["Stoch_D"].clip(0, 100)

    return data


def add_williams_r(df: pd.DataFrame, window: int = STOCH_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    high_max = data["High"].rolling(window=window, min_periods=max(5, window // 2)).max()
    low_min = data["Low"].rolling(window=window, min_periods=max(5, window // 2)).min()

    denominator = (high_max - low_min).replace(0, np.nan)

    data["Williams_R"] = -100 * (high_max - data["Close"]) / denominator
    data["Williams_R"] = data["Williams_R"].clip(-100, 0)

    return data


def add_cci(df: pd.DataFrame, window: int = CCI_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    sma_tp = typical_price.rolling(window=window, min_periods=max(5, window // 2)).mean()

    mean_deviation = typical_price.rolling(window=window, min_periods=max(5, window // 2)).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    data["CCI"] = (typical_price - sma_tp) / (0.015 * mean_deviation.replace(0, np.nan))
    data["CCI"] = data["CCI"].replace([np.inf, -np.inf], np.nan)

    return data


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    close = clean_numeric_series(data["Close"])
    volume = clean_numeric_series(data["Volume"])

    direction = np.sign(close.diff()).fillna(0)
    data["OBV"] = (direction * volume).cumsum()
    data["OBV_MA_20"] = data["OBV"].rolling(window=20, min_periods=5).mean()

    data["OBV_Trend"] = np.where(data["OBV"] > data["OBV_MA_20"], 1, -1)

    return data


def add_vwap_approx(df: pd.DataFrame, window: int = VWAP_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    volume = clean_numeric_series(data["Volume"])

    pv = typical_price * volume

    rolling_pv = pv.rolling(window=window, min_periods=max(5, window // 2)).sum()
    rolling_volume = volume.rolling(window=window, min_periods=max(5, window // 2)).sum()

    data["VWAP_Approx"] = rolling_pv / rolling_volume.replace(0, np.nan)
    data["Close_Above_VWAP"] = (data["Close"] > data["VWAP_Approx"]).astype(int)

    return data


def add_donchian_channel(df: pd.DataFrame, window: int = DONCHIAN_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    data["Donchian_High"] = data["High"].rolling(window=window, min_periods=max(5, window // 2)).max()
    data["Donchian_Low"] = data["Low"].rolling(window=window, min_periods=max(5, window // 2)).min()
    data["Donchian_Mid"] = (data["Donchian_High"] + data["Donchian_Low"]) / 2

    channel_width = (data["Donchian_High"] - data["Donchian_Low"]).replace(0, np.nan)

    data["Donchian_Position"] = (data["Close"] - data["Donchian_Low"]) / channel_width
    data["Donchian_Position"] = data["Donchian_Position"].clip(0, 1)

    data["Donchian_Breakout_Up"] = (data["Close"] >= data["Donchian_High"].shift(1)).astype(int)
    data["Donchian_Breakout_Down"] = (data["Close"] <= data["Donchian_Low"].shift(1)).astype(int)

    return data


def add_support_resistance(df: pd.DataFrame, window: int = SUPPORT_RESISTANCE_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    data["Rolling_Support"] = data["Low"].rolling(window=window, min_periods=max(5, window // 2)).min()
    data["Rolling_Resistance"] = data["High"].rolling(window=window, min_periods=max(5, window // 2)).max()

    close = data["Close"].replace(0, np.nan)

    data["Distance_To_Support"] = (data["Close"] - data["Rolling_Support"]) / close
    data["Distance_To_Resistance"] = (data["Rolling_Resistance"] - data["Close"]) / close

    data["Near_Support"] = (data["Distance_To_Support"].abs() <= 0.025).astype(int)
    data["Near_Resistance"] = (data["Distance_To_Resistance"].abs() <= 0.025).astype(int)

    return data


def add_bollinger_squeeze(df: pd.DataFrame, window: int = BOLLINGER_SQUEEZE_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    if "BB_Width" not in data.columns:
        close = clean_numeric_series(data["Close"])
        middle = close.rolling(window=20, min_periods=5).mean()
        std = close.rolling(window=20, min_periods=5).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        data["BB_Width"] = (upper - lower) / middle.replace(0, np.nan)

    data["BB_Width_MA"] = data["BB_Width"].rolling(window=window, min_periods=max(5, window // 2)).mean()
    data["BB_Squeeze"] = (data["BB_Width"] < data["BB_Width_MA"] * 0.75).astype(int)

    return data


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    data = df.copy()

    if "Annualized_Volatility" in data.columns:
        vol = clean_numeric_series(data["Annualized_Volatility"])
    elif "Volatility_20D" in data.columns:
        vol = clean_numeric_series(data["Volatility_20D"]) * np.sqrt(252)
    else:
        returns = data["Close"].pct_change()
        vol = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)

    vol_mean = vol.rolling(window=60, min_periods=20).mean()
    vol_std = vol.rolling(window=60, min_periods=20).std()

    data["Advanced_Volatility"] = vol
    data["Volatility_Regime_Z"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    conditions = [
        data["Volatility_Regime_Z"] >= 1.0,
        data["Volatility_Regime_Z"] <= -1.0,
    ]

    choices = ["high", "low"]

    data["Volatility_Regime"] = np.select(conditions, choices, default="normal")

    return data


def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = clean_price_data(df)

    if data.empty:
        return _empty_advanced_frame()

    data = add_fibonacci_levels(data)
    data = add_adx(data)
    data = add_stochastic(data)
    data = add_williams_r(data)
    data = add_cci(data)
    data = add_obv(data)
    data = add_vwap_approx(data)
    data = add_donchian_channel(data)
    data = add_support_resistance(data)
    data = add_bollinger_squeeze(data)
    data = add_volatility_regime(data)

    return data


def get_latest_advanced_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    data = add_advanced_indicators(df)

    if data.empty:
        return {
            "available": False,
            "message": "Advanced indicators unavailable.",
        }

    latest = data.dropna(subset=["Close"]).iloc[-1]

    close = safe_float(latest.get("Close"), 0.0)
    nearest_fib_level = safe_float(latest.get("Nearest_Fib_Level"), 0.0)
    nearest_fib_distance = safe_float(latest.get("Nearest_Fib_Distance"), 1.0)

    fib_zone = "neutral"

    if nearest_fib_distance <= 0.025:
        if close >= nearest_fib_level:
            fib_zone = "support"
        else:
            fib_zone = "resistance"

    return {
        "available": True,
        "close": close,
        "fib_high": safe_float(latest.get("Fib_High"), 0.0),
        "fib_low": safe_float(latest.get("Fib_Low"), 0.0),
        "fib_236": safe_float(latest.get("Fib_236"), 0.0),
        "fib_382": safe_float(latest.get("Fib_382"), 0.0),
        "fib_500": safe_float(latest.get("Fib_500"), 0.0),
        "fib_618": safe_float(latest.get("Fib_618"), 0.0),
        "fib_786": safe_float(latest.get("Fib_786"), 0.0),
        "nearest_fib_level": nearest_fib_level,
        "nearest_fib_name": str(latest.get("Nearest_Fib_Name", "")),
        "nearest_fib_distance": nearest_fib_distance,
        "fib_zone": fib_zone,
        "adx": safe_float(latest.get("ADX"), 0.0),
        "plus_di": safe_float(latest.get("Plus_DI"), 0.0),
        "minus_di": safe_float(latest.get("Minus_DI"), 0.0),
        "adx_bullish": int(safe_float(latest.get("ADX_Bullish"), 0)),
        "stoch_k": safe_float(latest.get("Stoch_K"), 50.0),
        "stoch_d": safe_float(latest.get("Stoch_D"), 50.0),
        "williams_r": safe_float(latest.get("Williams_R"), -50.0),
        "cci": safe_float(latest.get("CCI"), 0.0),
        "obv": safe_float(latest.get("OBV"), 0.0),
        "obv_ma_20": safe_float(latest.get("OBV_MA_20"), 0.0),
        "obv_trend": int(safe_float(latest.get("OBV_Trend"), 0)),
        "vwap_approx": safe_float(latest.get("VWAP_Approx"), 0.0),
        "close_above_vwap": int(safe_float(latest.get("Close_Above_VWAP"), 0)),
        "donchian_high": safe_float(latest.get("Donchian_High"), 0.0),
        "donchian_low": safe_float(latest.get("Donchian_Low"), 0.0),
        "donchian_mid": safe_float(latest.get("Donchian_Mid"), 0.0),
        "donchian_position": safe_float(latest.get("Donchian_Position"), 0.5),
        "donchian_breakout_up": int(safe_float(latest.get("Donchian_Breakout_Up"), 0)),
        "donchian_breakout_down": int(safe_float(latest.get("Donchian_Breakout_Down"), 0)),
        "rolling_support": safe_float(latest.get("Rolling_Support"), 0.0),
        "rolling_resistance": safe_float(latest.get("Rolling_Resistance"), 0.0),
        "distance_to_support": safe_float(latest.get("Distance_To_Support"), 0.0),
        "distance_to_resistance": safe_float(latest.get("Distance_To_Resistance"), 0.0),
        "near_support": int(safe_float(latest.get("Near_Support"), 0)),
        "near_resistance": int(safe_float(latest.get("Near_Resistance"), 0)),
        "bb_squeeze": int(safe_float(latest.get("BB_Squeeze"), 0)),
        "advanced_volatility": safe_float(latest.get("Advanced_Volatility"), 0.0),
        "volatility_regime_z": safe_float(latest.get("Volatility_Regime_Z"), 0.0),
        "volatility_regime": str(latest.get("Volatility_Regime", "normal")),
    }


def build_fibonacci_table(df: pd.DataFrame) -> pd.DataFrame:
    advanced = get_latest_advanced_indicators(df)

    if not advanced.get("available"):
        return pd.DataFrame(columns=["Level", "Price"])

    rows = [
        {"Level": "High", "Price": advanced["fib_high"]},
        {"Level": "23.6%", "Price": advanced["fib_236"]},
        {"Level": "38.2%", "Price": advanced["fib_382"]},
        {"Level": "50.0%", "Price": advanced["fib_500"]},
        {"Level": "61.8%", "Price": advanced["fib_618"]},
        {"Level": "78.6%", "Price": advanced["fib_786"]},
        {"Level": "Low", "Price": advanced["fib_low"]},
    ]

    return pd.DataFrame(rows)


def summarize_advanced_indicator_state(df: pd.DataFrame) -> Dict[str, Any]:
    advanced = get_latest_advanced_indicators(df)

    if not advanced.get("available"):
        return advanced

    summary = {
        "available": True,
        "fib_zone": advanced["fib_zone"],
        "nearest_fib_name": advanced["nearest_fib_name"],
        "nearest_fib_distance": advanced["nearest_fib_distance"],
        "adx_strength": "strong" if advanced["adx"] >= 25 else "weak",
        "adx_direction": "bullish" if advanced["adx_bullish"] else "bearish",
        "stochastic_zone": "overbought" if advanced["stoch_k"] >= 80 else "oversold" if advanced["stoch_k"] <= 20 else "neutral",
        "williams_zone": "overbought" if advanced["williams_r"] >= -20 else "oversold" if advanced["williams_r"] <= -80 else "neutral",
        "cci_zone": "bullish" if advanced["cci"] >= 100 else "bearish" if advanced["cci"] <= -100 else "neutral",
        "obv_trend": "bullish" if advanced["obv_trend"] > 0 else "bearish",
        "vwap_state": "above_vwap" if advanced["close_above_vwap"] else "below_vwap",
        "donchian_state": "breakout_up" if advanced["donchian_breakout_up"] else "breakout_down" if advanced["donchian_breakout_down"] else "inside_channel",
        "support_resistance_state": "near_support" if advanced["near_support"] else "near_resistance" if advanced["near_resistance"] else "neutral",
        "bollinger_squeeze": bool(advanced["bb_squeeze"]),
        "volatility_regime": advanced["volatility_regime"],
    }

    return summary