from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import DEFAULT_INTERVAL, DEFAULT_PERIOD
from src.utils import (
    clean_numeric_series,
    logger,
    safe_float,
    safe_pct_change,
    standardize_ohlcv_columns,
    ticker_clean,
)


REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Return",
            "Log_Return",
        ]
    )


def validate_ticker(ticker: Any) -> str:
    ticker = ticker_clean(ticker)

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    return ticker


def clean_price_data(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_price_frame()

    df = standardize_ohlcv_columns(df)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if df.empty:
        return _empty_price_frame()

    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()

    for col in REQUIRED_OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_cols:
        df[col] = clean_numeric_series(df[col])

    df = df.dropna(subset=["Close"])

    if df.empty:
        return _empty_price_frame()

    df["Volume"] = df["Volume"].fillna(0.0)
    df["Return"] = safe_pct_change(df["Close"])
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Log_Return"] = df["Log_Return"].replace([np.inf, -np.inf], np.nan)

    return df


def get_price_data(
    ticker: Any,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    ticker = validate_ticker(ticker)

    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )

        df = clean_price_data(df)

        if df.empty:
            logger.warning("No price data found for ticker %s", ticker)

        return df

    except Exception as error:
        logger.exception("Failed to download price data for %s: %s", ticker, error)
        return _empty_price_frame()


def get_ticker_info(ticker: Any) -> Dict[str, Any]:
    ticker = validate_ticker(ticker)

    try:
        info = yf.Ticker(ticker).info or {}

        return {
            "ticker": ticker,
            "short_name": info.get("shortName") or info.get("longName") or ticker,
            "long_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "market_cap": safe_float(info.get("marketCap"), 0.0),
            "currency": info.get("currency") or "",
            "exchange": info.get("exchange") or "",
            "country": info.get("country") or "",
        }

    except Exception as error:
        logger.warning("Could not fetch ticker info for %s: %s", ticker, error)

        return {
            "ticker": ticker,
            "short_name": ticker,
            "long_name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0.0,
            "currency": "",
            "exchange": "",
            "country": "",
        }


def get_latest_price_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "latest_close": 0.0,
            "previous_close": 0.0,
            "daily_return": 0.0,
            "volume": 0.0,
            "data_points": 0,
            "start_date": "",
            "end_date": "",
        }

    df = clean_price_data(df)

    if df.empty:
        return {
            "latest_close": 0.0,
            "previous_close": 0.0,
            "daily_return": 0.0,
            "volume": 0.0,
            "data_points": 0,
            "start_date": "",
            "end_date": "",
        }

    latest_close = safe_float(df["Close"].iloc[-1], 0.0)

    if len(df) >= 2:
        previous_close = safe_float(df["Close"].iloc[-2], latest_close)
    else:
        previous_close = latest_close

    daily_return = 0.0

    if previous_close:
        daily_return = (latest_close / previous_close) - 1.0

    volume = safe_float(df["Volume"].iloc[-1], 0.0)

    return {
        "latest_close": round(latest_close, 4),
        "previous_close": round(previous_close, 4),
        "daily_return": round(daily_return, 6),
        "volume": round(volume, 2),
        "data_points": int(len(df)),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
    }


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_price_frame()

    df = clean_price_data(df)

    if df.empty:
        return df

    df["Return_1D"] = safe_pct_change(df["Close"], 1)
    df["Return_3D"] = safe_pct_change(df["Close"], 3)
    df["Return_5D"] = safe_pct_change(df["Close"], 5)
    df["Return_10D"] = safe_pct_change(df["Close"], 10)
    df["Return_20D"] = safe_pct_change(df["Close"], 20)

    df["Forward_Return_1D"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["Forward_Return_3D"] = df["Close"].shift(-3) / df["Close"] - 1.0
    df["Forward_Return_5D"] = df["Close"].shift(-5) / df["Close"] - 1.0

    return df


def get_market_snapshot(
    ticker: Any,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> Dict[str, Any]:
    ticker = validate_ticker(ticker)

    price_df = get_price_data(ticker=ticker, period=period, interval=interval)
    info = get_ticker_info(ticker)
    summary = get_latest_price_summary(price_df)

    return {
        "ticker": ticker,
        "info": info,
        "summary": summary,
        "price_data": price_df,
        "success": not price_df.empty,
    }