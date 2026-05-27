import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import DEFAULT_INTERVAL, DEFAULT_PERIOD, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.indicators import add_all_indicators
from src.market_data import get_price_data
from src.scraper import fetch_all_news
from src.sentiment import aggregate_sentiment, analyze_news_batch
from src.utils import (
    clean_text,
    logger,
    safe_float,
    safe_str,
    ticker_clean,
)


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "TATASTEEL.NS",
    "SBIN.NS",
]


FEATURE_COLUMNS = [
    "ticker",
    "date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Return",
    "Log_Return",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ATR",
    "ATR_Pct",
    "BB_Width",
    "BB_Position",
    "Volume_MA_20",
    "Volume_Spike",
    "Volume_ZScore",
    "Volatility_20D",
    "Annualized_Volatility",
    "Volatility_ZScore",
    "Daily_Range",
    "Gap",
    "Close_Position",
    "Distance_SMA20",
    "Distance_SMA50",
    "Distance_SMA200",
    "Close_Above_SMA20",
    "Close_Above_SMA50",
    "Close_Above_SMA200",
    "MACD_Bullish",
    "sentiment_score",
    "average_confidence",
    "news_count",
    "positive_count",
    "negative_count",
    "neutral_count",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "dominant_label",
]


def ensure_training_dirs() -> None:
    (RAW_DATA_DIR / "prices").mkdir(parents=True, exist_ok=True)
    (RAW_DATA_DIR / "news").mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_ticker_for_file(ticker: Any) -> str:
    ticker = ticker_clean(ticker)
    return ticker.replace(".", "_").replace("-", "_").replace("=", "_")


def read_tickers_from_file(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_TICKERS

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {file_path}")

    tickers = []

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

        if "ticker" in df.columns:
            tickers = df["ticker"].dropna().astype(str).tolist()
        else:
            tickers = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        tickers = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    tickers = [ticker_clean(ticker) for ticker in tickers if ticker_clean(ticker)]

    return sorted(list(set(tickers)))


def load_cached_news(ticker: Any) -> pd.DataFrame:
    ticker = ticker_clean(ticker)
    safe_ticker = normalize_ticker_for_file(ticker)

    candidate_paths = [
        RAW_DATA_DIR / "news" / f"{safe_ticker}_news.csv",
        RAW_DATA_DIR / "news" / f"{ticker}_news.csv",
        RAW_DATA_DIR / "news" / f"{ticker.replace('.NS', '')}_news.csv",
        RAW_DATA_DIR / "news" / f"{ticker.replace('.BO', '')}_news.csv",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df
            except Exception as error:
                logger.warning("Could not read cached news file %s: %s", path, error)

    return pd.DataFrame()


def standardize_news_dates(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame()

    df = news_df.copy()

    if "published_at" not in df.columns:
        df["published_at"] = pd.NaT

    if "headline" not in df.columns:
        df["headline"] = ""

    df["headline"] = df["headline"].apply(clean_text)

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True,
    )

    if "scraped_at" in df.columns:
        fallback_dates = pd.to_datetime(
            df["scraped_at"],
            errors="coerce",
            utc=True,
        )
        df["published_at"] = df["published_at"].fillna(fallback_dates)

    df = df[df["headline"].str.len() > 0].copy()

    if df.empty:
        return pd.DataFrame()

    df["date"] = df["published_at"].dt.tz_convert("UTC").dt.date.astype(str)

    return df


def score_news_daily(
    news_df: pd.DataFrame,
    use_sentiment_model: bool = True,
) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "sentiment_score",
                "average_confidence",
                "news_count",
                "positive_count",
                "negative_count",
                "neutral_count",
                "positive_ratio",
                "negative_ratio",
                "neutral_ratio",
                "dominant_label",
            ]
        )

    df = standardize_news_dates(news_df)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for date, group in df.groupby("date"):
        headlines = group["headline"].dropna().astype(str).tolist()

        if use_sentiment_model:
            sentiment_df = analyze_news_batch(headlines)
            summary = aggregate_sentiment(sentiment_df)
        else:
            summary = {
                "sentiment_score": 0.0,
                "average_confidence": 0.0,
                "news_count": len(headlines),
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": len(headlines),
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 1.0,
                "dominant_label": "neutral",
            }

        summary["date"] = date
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def make_empty_sentiment_frame(price_dates: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": list(price_dates),
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
    )


def build_price_features(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    price_df = get_price_data(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    if price_df is None or price_df.empty:
        logger.warning("Skipping %s because price data is empty.", ticker)
        return pd.DataFrame()

    feature_df = add_all_indicators(price_df)

    if feature_df is None or feature_df.empty:
        logger.warning("Skipping %s because indicator data is empty.", ticker)
        return pd.DataFrame()

    feature_df = feature_df.copy()
    feature_df = feature_df.reset_index()

    first_col = feature_df.columns[0]

    if first_col != "date":
        feature_df = feature_df.rename(columns={first_col: "date"})

    feature_df["date"] = pd.to_datetime(
        feature_df["date"],
        errors="coerce",
        utc=True,
    ).dt.date.astype(str)

    feature_df["ticker"] = ticker_clean(ticker)

    return feature_df


def build_news_features(
    ticker: str,
    company_name: str = "",
    max_news: int = 50,
    fetch_live_news: bool = False,
    use_sentiment_model: bool = True,
) -> pd.DataFrame:
    news_df = load_cached_news(ticker)

    if fetch_live_news:
        live_news_df = fetch_all_news(
            ticker=ticker,
            company_name=company_name,
            max_items=max_news,
            save_cache=True,
        )

        if live_news_df is not None and not live_news_df.empty:
            if news_df is not None and not news_df.empty:
                news_df = pd.concat([news_df, live_news_df], ignore_index=True)
            else:
                news_df = live_news_df

    if news_df is None or news_df.empty:
        return pd.DataFrame()

    if "ticker" not in news_df.columns:
        news_df["ticker"] = ticker

    news_df["ticker"] = news_df["ticker"].fillna(ticker).astype(str)

    daily_sentiment = score_news_daily(
        news_df=news_df,
        use_sentiment_model=use_sentiment_model,
    )

    return daily_sentiment


def merge_price_and_news_features(
    ticker: str,
    price_features: pd.DataFrame,
    news_features: pd.DataFrame,
) -> pd.DataFrame:
    if price_features is None or price_features.empty:
        return pd.DataFrame()

    df = price_features.copy()

    if news_features is None or news_features.empty:
        news_features = make_empty_sentiment_frame(df["date"].tolist())

    merged = df.merge(
        news_features,
        on="date",
        how="left",
    )

    sentiment_defaults = {
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

    for col, default in sentiment_defaults.items():
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default)

    merged["ticker"] = ticker_clean(ticker)

    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[FEATURE_COLUMNS].copy()

    numeric_cols = [col for col in merged.columns if col not in {"ticker", "date", "dominant_label"}]

    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.replace([np.inf, -np.inf], np.nan)

    return merged


def build_dataset_for_ticker(
    ticker: str,
    company_name: str = "",
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    max_news: int = 50,
    fetch_live_news: bool = False,
    use_sentiment_model: bool = True,
) -> pd.DataFrame:
    ticker = ticker_clean(ticker)

    if not ticker:
        return pd.DataFrame()

    logger.info("Building dataset for %s", ticker)

    price_features = build_price_features(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    if price_features.empty:
        return pd.DataFrame()

    news_features = build_news_features(
        ticker=ticker,
        company_name=company_name,
        max_news=max_news,
        fetch_live_news=fetch_live_news,
        use_sentiment_model=use_sentiment_model,
    )

    merged = merge_price_and_news_features(
        ticker=ticker,
        price_features=price_features,
        news_features=news_features,
    )

    return merged


def build_full_dataset(
    tickers: Sequence[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    max_news: int = 50,
    fetch_live_news: bool = False,
    use_sentiment_model: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    ensure_training_dirs()

    frames = []

    for ticker in tickers:
        try:
            ticker_df = build_dataset_for_ticker(
                ticker=ticker,
                company_name="",
                period=period,
                interval=interval,
                max_news=max_news,
                fetch_live_news=fetch_live_news,
                use_sentiment_model=use_sentiment_model,
            )

            if ticker_df is not None and not ticker_df.empty:
                frames.append(ticker_df)
                logger.info("Added %s rows for %s", len(ticker_df), ticker)
            else:
                logger.warning("No rows created for %s", ticker)

        except Exception as error:
            logger.exception("Failed to build dataset for %s: %s", ticker, error)

    if not frames:
        logger.warning("No dataset rows were generated.")
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    dataset = pd.concat(frames, ignore_index=True)

    dataset = dataset.sort_values(["ticker", "date"]).reset_index(drop=True)

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "feature_dataset.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    logger.info("Saved feature dataset to %s", output_path)
    logger.info("Final dataset shape: %s", dataset.shape)

    return dataset


def build_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    if dataset is None or dataset.empty:
        return {
            "rows": 0,
            "tickers": 0,
            "start_date": "",
            "end_date": "",
            "avg_news_count": 0.0,
            "avg_sentiment": 0.0,
        }

    return {
        "rows": int(len(dataset)),
        "tickers": int(dataset["ticker"].nunique()) if "ticker" in dataset.columns else 0,
        "start_date": safe_str(dataset["date"].min()) if "date" in dataset.columns else "",
        "end_date": safe_str(dataset["date"].max()) if "date" in dataset.columns else "",
        "avg_news_count": round(safe_float(dataset["news_count"].mean(), 0.0), 4)
        if "news_count" in dataset.columns
        else 0.0,
        "avg_sentiment": round(safe_float(dataset["sentiment_score"].mean(), 0.0), 6)
        if "sentiment_score" in dataset.columns
        else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FinSentinel feature dataset from price, technical, and news sentiment data."
    )

    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Ticker list. Example: AAPL MSFT TATASTEEL.NS",
    )

    parser.add_argument(
        "--ticker-file",
        default=None,
        help="Path to txt/csv file containing tickers.",
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="yfinance period. Example: 6mo, 1y, 2y, 5y",
    )

    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        help="yfinance interval. Example: 1d, 1wk, 1mo",
    )

    parser.add_argument(
        "--max-news",
        type=int,
        default=50,
        help="Maximum news items per ticker if live fetching is enabled.",
    )

    parser.add_argument(
        "--fetch-live-news",
        action="store_true",
        help="Fetch live news using scraper before building dataset.",
    )

    parser.add_argument(
        "--no-sentiment-model",
        action="store_true",
        help="Disable FinBERT scoring and use neutral sentiment placeholders.",
    )

    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "feature_dataset.csv"),
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers:
        tickers = [ticker_clean(ticker) for ticker in args.tickers if ticker_clean(ticker)]
    else:
        tickers = read_tickers_from_file(args.ticker_file)

    tickers = sorted(list(set(tickers)))

    if not tickers:
        raise ValueError("No valid tickers provided.")

    dataset = build_full_dataset(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        max_news=args.max_news,
        fetch_live_news=args.fetch_live_news,
        use_sentiment_model=not args.no_sentiment_model,
        output_path=Path(args.output),
    )

    summary = build_summary(dataset)

    print("\nFinSentinel Dataset Build Summary")
    print("---------------------------------")
    print(f"Rows: {summary['rows']}")
    print(f"Tickers: {summary['tickers']}")
    print(f"Start date: {summary['start_date']}")
    print(f"End date: {summary['end_date']}")
    print(f"Average news count: {summary['avg_news_count']}")
    print(f"Average sentiment: {summary['avg_sentiment']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()