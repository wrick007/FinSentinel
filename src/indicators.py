from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import (
    SMA_SHORT_WINDOW,
    SMA_LONG_WINDOW,
    SMA_MAJOR_WINDOW,
    EMA_FAST_WINDOW,
    EMA_SLOW_WINDOW,
    MACD_SIGNAL_WINDOW,
    RSI_WINDOW,
    ATR_WINDOW,
    VOLATILITY_WINDOW,
    VOLUME_WINDOW,
    BOLLINGER_WINDOW,
    BOLLINGER_STD,
)
from src.market_data import clean_price_data
from src.utils import clean_numeric_series, logger, rolling_zscore, safe_float


def _empty_indicator_frame() -> pd.DataFrame:
    return pd.DataFrame()


def add_sma(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    df["SMA_20"] = df["Close"].rolling(
        window=SMA_SHORT_WINDOW,
        min_periods=max(3, SMA_SHORT_WINDOW // 3),
    ).mean()

    df["SMA_50"] = df["Close"].rolling(
        window=SMA_LONG_WINDOW,
        min_periods=max(5, SMA_LONG_WINDOW // 3),
    ).mean()

    df["SMA_200"] = df["Close"].rolling(
        window=SMA_MAJOR_WINDOW,
        min_periods=max(20, SMA_MAJOR_WINDOW // 4),
    ).mean()

    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    df["EMA_12"] = df["Close"].ewm(
        span=EMA_FAST_WINDOW,
        adjust=False,
        min_periods=max(3, EMA_FAST_WINDOW // 3),
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=EMA_SLOW_WINDOW,
        adjust=False,
        min_periods=max(5, EMA_SLOW_WINDOW // 3),
    ).mean()

    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    close = clean_numeric_series(df["Close"])
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi.clip(0, 100)
    df["RSI"] = df["RSI"].fillna(50)

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    close = clean_numeric_series(df["Close"])

    ema_fast = close.ewm(
        span=EMA_FAST_WINDOW,
        adjust=False,
        min_periods=max(3, EMA_FAST_WINDOW // 3),
    ).mean()

    ema_slow = close.ewm(
        span=EMA_SLOW_WINDOW,
        adjust=False,
        min_periods=max(5, EMA_SLOW_WINDOW // 3),
    ).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(
        span=MACD_SIGNAL_WINDOW,
        adjust=False,
        min_periods=max(3, MACD_SIGNAL_WINDOW // 3),
    ).mean()

    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd - macd_signal

    return df


def add_atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    high = clean_numeric_series(df["High"])
    low = clean_numeric_series(df["Low"])
    close = clean_numeric_series(df["Close"])

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["TR"] = true_range
    df["ATR"] = true_range.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=max(3, window // 2),
    ).mean()

    df["ATR_Pct"] = df["ATR"] / close.replace(0, np.nan)
    df["ATR_Pct"] = df["ATR_Pct"].replace([np.inf, -np.inf], np.nan)

    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    close = clean_numeric_series(df["Close"])

    middle = close.rolling(
        window=BOLLINGER_WINDOW,
        min_periods=max(5, BOLLINGER_WINDOW // 3),
    ).mean()

    std = close.rolling(
        window=BOLLINGER_WINDOW,
        min_periods=max(5, BOLLINGER_WINDOW // 3),
    ).std()

    upper = middle + (BOLLINGER_STD * std)
    lower = middle - (BOLLINGER_STD * std)

    df["BB_Middle"] = middle
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = (upper - lower) / middle.replace(0, np.nan)
    df["BB_Position"] = (close - lower) / (upper - lower).replace(0, np.nan)

    df["BB_Width"] = df["BB_Width"].replace([np.inf, -np.inf], np.nan)
    df["BB_Position"] = df["BB_Position"].replace([np.inf, -np.inf], np.nan)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    volume = clean_numeric_series(df["Volume"], fill_value=0.0)
    close = clean_numeric_series(df["Close"])

    df["Volume_MA_20"] = volume.rolling(
        window=VOLUME_WINDOW,
        min_periods=max(3, VOLUME_WINDOW // 3),
    ).mean()

    df["Volume_Spike"] = volume / df["Volume_MA_20"].replace(0, np.nan)
    df["Volume_Spike"] = df["Volume_Spike"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()

    df["Volume_ZScore"] = rolling_zscore(volume, window=VOLUME_WINDOW)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    returns = clean_numeric_series(df["Return"])

    df["Volatility_20D"] = returns.rolling(
        window=VOLATILITY_WINDOW,
        min_periods=max(5, VOLATILITY_WINDOW // 3),
    ).std()

    df["Annualized_Volatility"] = df["Volatility_20D"] * np.sqrt(252)
    df["Volatility_ZScore"] = rolling_zscore(df["Volatility_20D"], window=VOLATILITY_WINDOW)

    return df


def add_price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    close = clean_numeric_series(df["Close"])
    open_price = clean_numeric_series(df["Open"])
    high = clean_numeric_series(df["High"])
    low = clean_numeric_series(df["Low"])

    df["Daily_Range"] = (high - low) / close.replace(0, np.nan)
    df["Gap"] = open_price / close.shift(1) - 1.0
    df["Close_Position"] = (close - low) / (high - low).replace(0, np.nan)

    df["Distance_SMA20"] = np.nan
    df["Distance_SMA50"] = np.nan
    df["Distance_SMA200"] = np.nan

    if "SMA_20" in df.columns:
        df["Distance_SMA20"] = close / df["SMA_20"].replace(0, np.nan) - 1.0

    if "SMA_50" in df.columns:
        df["Distance_SMA50"] = close / df["SMA_50"].replace(0, np.nan) - 1.0

    if "SMA_200" in df.columns:
        df["Distance_SMA200"] = close / df["SMA_200"].replace(0, np.nan) - 1.0

    numeric_cols = [
        "Daily_Range",
        "Gap",
        "Close_Position",
        "Distance_SMA20",
        "Distance_SMA50",
        "Distance_SMA200",
    ]

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_price_data(df)

    if df.empty:
        return df

    close = clean_numeric_series(df["Close"])

    if "SMA_20" in df.columns:
        df["Close_Above_SMA20"] = (close > df["SMA_20"]).astype(int)
    else:
        df["Close_Above_SMA20"] = 0

    if "SMA_50" in df.columns:
        df["Close_Above_SMA50"] = (close > df["SMA_50"]).astype(int)
    else:
        df["Close_Above_SMA50"] = 0

    if "SMA_200" in df.columns:
        df["Close_Above_SMA200"] = (close > df["SMA_200"]).astype(int)
    else:
        df["Close_Above_SMA200"] = 0

    if {"SMA_20", "SMA_50"}.issubset(df.columns):
        df["SMA20_Above_SMA50"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    else:
        df["SMA20_Above_SMA50"] = 0

    if {"MACD", "MACD_Signal"}.issubset(df.columns):
        df["MACD_Bullish"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
    else:
        df["MACD_Bullish"] = 0

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = clean_price_data(df)

        if df.empty:
            return df

        df = add_sma(df)
        df = add_ema(df)
        df = add_rsi(df)
        df = add_macd(df)
        df = add_atr(df)
        df = add_bollinger_bands(df)
        df = add_volume_features(df)
        df = add_volatility(df)
        df = add_price_structure_features(df)
        df = add_trend_flags(df)

        required_defaults = {
            "SMA_20": np.nan,
            "SMA_50": np.nan,
            "SMA_200": np.nan,
            "EMA_12": np.nan,
            "EMA_26": np.nan,
            "RSI": 50.0,
            "MACD": 0.0,
            "MACD_Signal": 0.0,
            "MACD_Hist": 0.0,
            "ATR": 0.0,
            "ATR_Pct": 0.0,
            "Volume_MA_20": np.nan,
            "Volume_Spike": 1.0,
            "OBV": 0.0,
            "Volume_ZScore": 0.0,
            "Volatility_20D": 0.0,
            "Annualized_Volatility": 0.0,
            "Volatility_ZScore": 0.0,
            "Gap": 0.0,
            "Distance_SMA20": 0.0,
            "Distance_SMA50": 0.0,
            "Distance_SMA200": 0.0,
            "Close_Above_SMA20": 0,
            "Close_Above_SMA50": 0,
            "Close_Above_SMA200": 0,
            "MACD_Bullish": 0,
            "BB_Position": 0.5,
        }

        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        df = df.replace([np.inf, -np.inf], np.nan)

        df["RSI"] = df["RSI"].fillna(50.0)
        df["Volume_Spike"] = df["Volume_Spike"].fillna(1.0)
        df["MACD"] = df["MACD"].fillna(0.0)
        df["MACD_Signal"] = df["MACD_Signal"].fillna(0.0)
        df["MACD_Hist"] = df["MACD_Hist"].fillna(0.0)
        df["Volatility_20D"] = df["Volatility_20D"].fillna(0.0)
        df["Annualized_Volatility"] = df["Annualized_Volatility"].fillna(0.0)

        return df

    except Exception as error:
        logger.exception("Failed to add technical indicators: %s", error)

        df = clean_price_data(df)

        if df.empty:
            return df

        fallback_columns = {
            "SMA_20": np.nan,
            "SMA_50": np.nan,
            "RSI": 50.0,
            "MACD": 0.0,
            "Volume_Spike": 1.0,
        }

        for col, default in fallback_columns.items():
            if col not in df.columns:
                df[col] = default

        return df

def get_latest_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    df = add_all_indicators(df)

    if df.empty:
        return {
            "available": False,
            "close": 0.0,
            "sma20": 0.0,
            "sma50": 0.0,
            "sma200": 0.0,
            "rsi": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_hist": 0.0,
            "atr_pct": 0.0,
            "volatility": 0.0,
            "volume_spike": 1.0,
            "gap": 0.0,
        }

    latest = df.iloc[-1]

    return {
        "available": True,
        "close": safe_float(latest.get("Close"), 0.0),
        "sma20": safe_float(latest.get("SMA_20"), 0.0),
        "sma50": safe_float(latest.get("SMA_50"), 0.0),
        "sma200": safe_float(latest.get("SMA_200"), 0.0),
        "rsi": safe_float(latest.get("RSI"), 50.0),
        "macd": safe_float(latest.get("MACD"), 0.0),
        "macd_signal": safe_float(latest.get("MACD_Signal"), 0.0),
        "macd_hist": safe_float(latest.get("MACD_Hist"), 0.0),
        "atr_pct": safe_float(latest.get("ATR_Pct"), 0.0),
        "volatility": safe_float(latest.get("Volatility_20D"), 0.0),
        "annualized_volatility": safe_float(latest.get("Annualized_Volatility"), 0.0),
        "volume_spike": safe_float(latest.get("Volume_Spike"), 1.0),
        "gap": safe_float(latest.get("Gap"), 0.0),
        "close_above_sma20": int(safe_float(latest.get("Close_Above_SMA20"), 0)),
        "close_above_sma50": int(safe_float(latest.get("Close_Above_SMA50"), 0)),
        "close_above_sma200": int(safe_float(latest.get("Close_Above_SMA200"), 0)),
        "macd_bullish": int(safe_float(latest.get("MACD_Bullish"), 0)),
        "distance_sma20": safe_float(latest.get("Distance_SMA20"), 0.0),
        "distance_sma50": safe_float(latest.get("Distance_SMA50"), 0.0),
        "bb_position": safe_float(latest.get("BB_Position"), 0.5),
    }


def summarize_indicator_state(df: pd.DataFrame) -> Dict[str, Any]:
    latest = get_latest_indicators(df)

    if not latest.get("available"):
        return {
            "trend_state": "Unknown",
            "momentum_state": "Unknown",
            "volatility_state": "Unknown",
            "volume_state": "Unknown",
        }

    close = latest["close"]
    sma20 = latest["sma20"]
    sma50 = latest["sma50"]
    rsi = latest["rsi"]
    volatility = latest["volatility"]
    volume_spike = latest["volume_spike"]

    if close > sma20 > sma50:
        trend_state = "Bullish"
    elif close < sma20 < sma50:
        trend_state = "Bearish"
    else:
        trend_state = "Mixed"

    if rsi >= 70:
        momentum_state = "Overbought"
    elif rsi <= 30:
        momentum_state = "Oversold"
    elif latest["macd_bullish"]:
        momentum_state = "Positive"
    else:
        momentum_state = "Neutral"

    if volatility >= 0.04:
        volatility_state = "High"
    elif volatility <= 0.015:
        volatility_state = "Low"
    else:
        volatility_state = "Normal"

    if volume_spike >= 1.5:
        volume_state = "High volume"
    elif volume_spike <= 0.7:
        volume_state = "Low volume"
    else:
        volume_state = "Normal volume"

    return {
        "trend_state": trend_state,
        "momentum_state": momentum_state,
        "volatility_state": volatility_state,
        "volume_state": volume_state,
    }