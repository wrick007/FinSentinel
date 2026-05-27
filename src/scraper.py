from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    DEFAULT_SOURCE_WEIGHT,
    GOOGLE_NEWS_RSS_URL,
    MAX_NEWS_ITEMS,
    NEWS_LOOKBACK_DAYS,
    RAW_DATA_DIR,
    SOURCE_WEIGHTS,
)
from src.utils import (
    clean_text,
    company_clean,
    logger,
    normalize_text,
    safe_float,
    safe_str,
    text_hash,
    ticker_clean,
)


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)


NEWS_COLUMNS = [
    "ticker",
    "company_name",
    "headline",
    "summary",
    "source",
    "url",
    "published_at",
    "scraped_at",
    "relevance_score",
    "source_weight",
    "news_hash",
]


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def _safe_news_ticker(ticker: Any) -> str:
    ticker = ticker_clean(ticker)

    if ticker.endswith(".NS"):
        return ticker.replace(".NS", "")

    if ticker.endswith(".BO"):
        return ticker.replace(".BO", "")

    return ticker


def _safe_file_ticker(ticker: Any) -> str:
    ticker = safe_str(ticker).upper()
    ticker = ticker.replace(".", "_").replace("-", "_").replace("=", "_")
    ticker = "".join(ch for ch in ticker if ch.isalnum() or ch == "_")
    return ticker or "UNKNOWN"


def build_google_news_query(
    ticker: Any,
    company_name: Any = "",
    extra_terms: Optional[List[str]] = None,
) -> str:
    ticker = _safe_news_ticker(ticker)
    company_name = company_clean(company_name)

    terms = []

    if company_name:
        terms.append(f'"{company_name}"')

    if ticker:
        terms.append(ticker)

    if extra_terms:
        for term in extra_terms:
            term = clean_text(term)
            if term:
                terms.append(term)

    if not terms:
        raise ValueError("Ticker or company name is required for news search.")

    entity_query = " OR ".join(terms)

    return f"({entity_query}) stock OR shares OR earnings OR finance OR market"


def parse_datetime(value: Any) -> pd.Timestamp:
    value = safe_str(value)

    if not value:
        return pd.NaT

    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return pd.NaT

    return parsed


def get_source_weight(source: Any) -> float:
    source = normalize_text(source).replace(" ", "_")

    for key, weight in SOURCE_WEIGHTS.items():
        if key in source:
            return safe_float(weight, DEFAULT_SOURCE_WEIGHT)

    return DEFAULT_SOURCE_WEIGHT


def calculate_relevance_score(
    headline: Any,
    summary: Any = "",
    ticker: Any = "",
    company_name: Any = "",
) -> float:
    headline_norm = normalize_text(headline)
    summary_norm = normalize_text(summary)
    ticker_norm = normalize_text(_safe_news_ticker(ticker))
    company_norm = normalize_text(company_name)

    full_text = f"{headline_norm} {summary_norm}"

    score = 0.0

    if company_norm and company_norm in headline_norm:
        score += 3.0

    if ticker_norm and ticker_norm in headline_norm:
        score += 2.5

    if company_norm and company_norm in summary_norm:
        score += 1.5

    if ticker_norm and ticker_norm in summary_norm:
        score += 1.0

    finance_terms = [
        "stock",
        "shares",
        "earnings",
        "revenue",
        "profit",
        "loss",
        "market",
        "analyst",
        "upgrade",
        "downgrade",
        "guidance",
        "dividend",
        "merger",
        "acquisition",
        "results",
        "quarter",
        "forecast",
        "ipo",
        "sec",
        "filing",
        "nse",
        "bse",
        "price target",
        "brokerage",
    ]

    for term in finance_terms:
        if term in full_text:
            score += 0.25

    noise_terms = [
        "football",
        "cricket",
        "movie",
        "celebrity",
        "recipe",
        "weather",
        "astrology",
    ]

    for term in noise_terms:
        if term in full_text:
            score -= 1.0

    return round(max(0.0, min(score, 10.0)), 4)


def clean_news(
    news_df: Optional[pd.DataFrame],
    ticker: Any = "",
    company_name: Any = "",
    min_relevance: float = 0.5,
) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return _empty_news_frame()

    df = news_df.copy()

    ticker = _safe_news_ticker(ticker)
    company_name = company_clean(company_name)

    required_columns = [
        "ticker",
        "company_name",
        "headline",
        "summary",
        "source",
        "url",
        "published_at",
        "scraped_at",
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = df["ticker"].apply(lambda x: _safe_news_ticker(x) or ticker)
    df["company_name"] = df["company_name"].apply(
        lambda x: company_clean(x) or company_name
    )
    df["headline"] = df["headline"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)
    df["source"] = df["source"].apply(lambda x: clean_text(x).lower() or "generic")
    df["url"] = df["url"].apply(safe_str)

    df = df[df["headline"].str.len() > 0].copy()

    if df.empty:
        return _empty_news_frame()

    df["published_at"] = pd.to_datetime(
        df["published_at"].apply(parse_datetime),
        errors="coerce",
        utc=True,
    )

    df["scraped_at"] = pd.to_datetime(
        df["scraped_at"],
        errors="coerce",
        utc=True,
    )

    df["scraped_at"] = df["scraped_at"].fillna(pd.Timestamp.now(tz="UTC"))

    now = pd.Timestamp.now(tz="UTC")
    earliest = now - pd.Timedelta(days=NEWS_LOOKBACK_DAYS)

    has_date = df["published_at"].notna()
    df = df[(~has_date) | (df["published_at"] >= earliest)].copy()

    if df.empty:
        return _empty_news_frame()

    df["news_hash"] = df.apply(
        lambda row: text_hash(f"{row.get('headline', '')} {row.get('url', '')}"),
        axis=1,
    )

    df = df.drop_duplicates(subset=["news_hash"]).copy()

    df["relevance_score"] = df.apply(
        lambda row: calculate_relevance_score(
            headline=row.get("headline", ""),
            summary=row.get("summary", ""),
            ticker=row.get("ticker", ticker),
            company_name=row.get("company_name", company_name),
        ),
        axis=1,
    )

    df = df[df["relevance_score"] >= min_relevance].copy()

    if df.empty:
        return _empty_news_frame()

    df["source_weight"] = df["source"].apply(get_source_weight)

    df = df.sort_values(
        by=["published_at", "relevance_score"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return df[NEWS_COLUMNS]


def fetch_google_news_rss(
    ticker: Any,
    company_name: Any = "",
    max_items: int = MAX_NEWS_ITEMS,
) -> pd.DataFrame:
    ticker = _safe_news_ticker(ticker)
    company_name = company_clean(company_name)

    try:
        query = build_google_news_query(ticker, company_name)
        encoded_query = quote_plus(query)

        url = (
            f"{GOOGLE_NEWS_RSS_URL}"
            f"?q={encoded_query}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )

        feed = feedparser.parse(url)

        rows = []
        scraped_at = datetime.now(timezone.utc).isoformat()

        for entry in feed.entries[: int(max_items)]:
            headline = clean_text(entry.get("title", ""))
            summary = clean_text(entry.get("summary", ""))
            link = safe_str(entry.get("link", ""))

            published_raw = (
                entry.get("published")
                or entry.get("updated")
                or entry.get("created")
                or ""
            )

            source = "google_news"

            try:
                entry_source = getattr(entry, "source", None)
                if isinstance(entry_source, dict):
                    source = clean_text(entry_source.get("title", "google_news")).lower()
            except Exception:
                source = "google_news"

            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "headline": headline,
                    "summary": summary,
                    "source": source,
                    "url": link,
                    "published_at": published_raw,
                    "scraped_at": scraped_at,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            return _empty_news_frame()

        return clean_news(
            news_df=df,
            ticker=ticker,
            company_name=company_name,
            min_relevance=0.5,
        )

    except Exception as error:
        logger.warning("Google News RSS fetch failed for %s: %s", ticker, error)
        return _empty_news_frame()


def fetch_yahoo_finance_news(
    ticker: Any,
    max_items: int = MAX_NEWS_ITEMS,
) -> pd.DataFrame:
    ticker = ticker_clean(ticker)

    if not ticker:
        return _empty_news_frame()

    url = f"https://finance.yahoo.com/quote/{ticker}/news"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        rows = []
        scraped_at = datetime.now(timezone.utc).isoformat()

        for link in soup.find_all("a", href=True):
            text = clean_text(link.get_text(" "))
            href = safe_str(link.get("href"))

            if len(text) < 25:
                continue

            if "/news/" not in href and "finance.yahoo.com" not in href:
                continue

            if href.startswith("/"):
                href = "https://finance.yahoo.com" + href

            rows.append(
                {
                    "ticker": _safe_news_ticker(ticker),
                    "company_name": "",
                    "headline": text,
                    "summary": "",
                    "source": "yahoo_finance",
                    "url": href,
                    "published_at": "",
                    "scraped_at": scraped_at,
                }
            )

            if len(rows) >= int(max_items):
                break

        df = pd.DataFrame(rows)

        if df.empty:
            return _empty_news_frame()

        return clean_news(
            news_df=df,
            ticker=_safe_news_ticker(ticker),
            company_name="",
            min_relevance=0.5,
        )

    except Exception as error:
        logger.warning("Yahoo Finance news fetch failed for %s: %s", ticker, error)
        return _empty_news_frame()


def save_news_cache(news_df: pd.DataFrame, ticker: Any) -> Optional[Path]:
    if news_df is None or news_df.empty:
        return None

    try:
        news_dir = RAW_DATA_DIR / "news"
        news_dir.mkdir(parents=True, exist_ok=True)

        safe_ticker = _safe_file_ticker(ticker)
        file_path = news_dir / f"{safe_ticker}_news.csv"

        news_df.to_csv(file_path, index=False)

        return file_path

    except Exception as error:
        logger.warning("Could not save news data: %s", error)
        return None


def fetch_all_news(
    ticker: Any,
    company_name: Any = "",
    max_items: int = MAX_NEWS_ITEMS,
    save_cache: bool = True,
) -> pd.DataFrame:
    ticker_original = ticker_clean(ticker)
    ticker_news = _safe_news_ticker(ticker_original)
    company_name = company_clean(company_name)

    if not ticker_news and not company_name:
        return _empty_news_frame()

    frames = []

    google_df = fetch_google_news_rss(
        ticker=ticker_news,
        company_name=company_name,
        max_items=max_items,
    )

    if google_df is not None and not google_df.empty:
        frames.append(google_df)

    yahoo_df = fetch_yahoo_finance_news(
        ticker=ticker_original,
        max_items=max_items,
    )

    if yahoo_df is not None and not yahoo_df.empty:
        yahoo_df["company_name"] = yahoo_df["company_name"].replace("", company_name)
        frames.append(yahoo_df)

    if not frames:
        return _empty_news_frame()

    combined = pd.concat(frames, ignore_index=True)

    combined = clean_news(
        news_df=combined,
        ticker=ticker_news,
        company_name=company_name,
        min_relevance=0.5,
    )

    if combined.empty:
        return _empty_news_frame()

    final_df = combined.head(int(max_items)).reset_index(drop=True)

    if save_cache:
        save_news_cache(final_df, ticker_news or ticker_original)

    return final_df


def news_dataframe_to_text_list(news_df: pd.DataFrame) -> List[str]:
    if news_df is None or news_df.empty or "headline" not in news_df.columns:
        return []

    return [
        clean_text(item)
        for item in news_df["headline"].tolist()
        if clean_text(item)
    ]


def build_news_display_table(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame(
            columns=[
                "Headline",
                "Source",
                "Published",
                "Relevance",
                "URL",
            ]
        )

    df = news_df.copy()

    for col in ["headline", "source", "published_at", "relevance_score", "url"]:
        if col not in df.columns:
            df[col] = ""

    display = pd.DataFrame(
        {
            "Headline": df["headline"].apply(clean_text),
            "Source": df["source"].apply(clean_text),
            "Published": df["published_at"].astype(str),
            "Relevance": pd.to_numeric(
                df["relevance_score"],
                errors="coerce",
            )
            .fillna(0.0)
            .round(2),
            "URL": df["url"].apply(safe_str),
        }
    )

    return display