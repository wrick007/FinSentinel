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

def news_dataframe_to_text_list(news_df):
    from src.utils import clean_text

    if news_df is None or news_df.empty or "headline" not in news_df.columns:
        return []

    return [
        clean_text(item)
        for item in news_df["headline"].tolist()
        if clean_text(item)
    ]


def build_news_display_table(news_df):
    import pandas as pd
    from src.utils import clean_text, safe_str

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
            "Relevance": pd.to_numeric(df["relevance_score"], errors="coerce")
            .fillna(0)
            .round(2),
            "URL": df["url"].apply(safe_str),
        }
    )

    return display

from datetime import datetime, timezone
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


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def build_google_news_query(
    ticker: Any,
    company_name: Any = "",
    extra_terms: Optional[List[str]] = None,
) -> str:
    ticker = ticker_clean(ticker)
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

    return pd.to_datetime(value, errors="coerce", utc=True)


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
    ticker_norm = normalize_text(ticker)
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

    ticker = ticker_clean(ticker)
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

    df["ticker"] = df["ticker"].apply(lambda x: ticker_clean(x) or ticker)
    df["company_name"] = df["company_name"].apply(lambda x: company_clean(x) or company_name)
    df["headline"] = df["headline"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)
    df["source"] = df["source"].apply(lambda x: clean_text(x).lower() or "generic")
    df["url"] = df["url"].apply(safe_str)

    df = df[df["headline"].str.len() > 0].copy()

    if df.empty:
        return _empty_news_frame()

    df["published_at"] = df["published_at"].apply(parse_datetime)
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)

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

    return df[
        [
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
    ]


def fetch_google_news_rss(
    ticker: Any,
    company_name: Any = "",
    max_items: int = MAX_NEWS_ITEMS,
) -> pd.DataFrame:
    ticker = ticker_clean(ticker)
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

        for entry in feed.entries[:max_items]:
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
                if hasattr(entry, "source") and isinstance(entry.source, dict):
                    source = clean_text(entry.source.get("title", "google_news")).lower()
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

        return clean_news(df, ticker=ticker, company_name=company_name)

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
                    "ticker": ticker,
                    "company_name": "",
                    "headline": text,
                    "summary": "",
                    "source": "yahoo_finance",
                    "url": href,
                    "published_at": "",
                    "scraped_at": scraped_at,
                }
            )

            if len(rows) >= max_items:
                break

        df = pd.DataFrame(rows)

        if df.empty:
            return _empty_news_frame()

        return clean_news(df, ticker=ticker, company_name="")

    except Exception as error:
        logger.warning("Yahoo Finance news fetch failed for %s: %s", ticker, error)
        return _empty_news_frame()


def fetch_all_news(
    ticker: Any,
    company_name: Any = "",
    max_items: int = MAX_NEWS_ITEMS,
) -> pd.DataFrame:
    ticker = ticker_clean(ticker)
    company_name = company_clean(company_name)

    if not ticker and not company_name:
        return _empty_news_frame()

    frames = []

    google_df = fetch_google_news_rss(
        ticker=ticker,
        company_name=company_name,
        max_items=max_items,
    )

    if google_df is not None and not google_df.empty:
        frames.append(google_df)

    yahoo_df = fetch_yahoo_finance_news(
        ticker=ticker,
        max_items=max_items,
    )

    if yahoo_df is not None and not yahoo_df.empty:
        yahoo_df["company_name"] = yahoo_df["company_name"].replace("", company_name)
        frames.append(yahoo_df)

    if not frames:
        return _empty_news_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = clean_news(
        combined,
        ticker=ticker,
        company_name=company_name,
        min_relevance=0.5,
    )

    if combined.empty:
        return _empty_news_frame()

    return combined.head(int(max_items)).reset_index(drop=True)


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
            ).fillna(0.0).round(2),
            "URL": df["url"].apply(safe_str),
        }
    )

    return display