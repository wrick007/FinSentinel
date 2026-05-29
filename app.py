import traceback
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_COMPANY_NAME,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_TICKER,
    STREAMLIT_LAYOUT,
    STREAMLIT_PAGE_ICON,
    STREAMLIT_PAGE_TITLE,
    SUPPORTED_INTERVALS,
    SUPPORTED_PERIODS,
)
from src.indicators import add_all_indicators
from src.market_data import get_price_data, get_ticker_info, get_latest_price_summary
from src.scraper import fetch_all_news, news_dataframe_to_text_list, build_news_display_table
from src.sentiment import analyze_news_batch, aggregate_sentiment
from src.signal_engine import (
    build_feature_contribution_table,
    build_signal_table,
    generate_signal,
)
from src.advanced_indicators import build_fibonacci_table
from src.indicator_signal_model import (
    build_advanced_reason_table,
    build_advanced_signal_table,
)
from src.risk_filters import build_risk_table
from src.ml_signal import build_ml_signal_table
from src.utils import clean_text, format_large_number, format_pct, safe_float, split_user_news_input, ticker_clean


DISCLAIMER = """
FinSentinel is an educational financial signal analysis tool. It is not financial advice.
Always verify information independently before making investment decisions.
"""


st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon=STREAMLIT_PAGE_ICON,
    layout=STREAMLIT_LAYOUT,
)

def safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    for col in result.columns:
        if result[col].dtype == "object":
            result[col] = result[col].astype(str)

    return result

def make_price_chart(price_df: pd.DataFrame, ticker: str, show_fibonacci: bool = True) -> go.Figure:
    if price_df is None or price_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Price chart unavailable", height=450)
        return fig

    df = add_all_indicators(price_df).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )

    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                mode="lines",
                name="SMA 20",
            )
        )

    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                mode="lines",
                name="SMA 50",
            )
        )

    if show_fibonacci:
        try:
            fib_table = build_fibonacci_table(df)

            for _, row in fib_table.iterrows():
                level = row.get("Level")
                price = safe_float(row.get("Price"), 0.0)

                if price > 0 and level not in {"High", "Low"}:
                    fig.add_hline(
                        y=price,
                        line_dash="dot",
                        annotation_text=f"Fib {level}",
                        annotation_position="right",
                    )

        except Exception:
            pass

    fig.update_layout(
        title=f"{ticker.upper()} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=30, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def make_rsi_chart(price_df: pd.DataFrame, ticker: str) -> go.Figure:
    df = add_all_indicators(price_df).copy()

    fig = go.Figure()

    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
            )
        )

    fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold")

    fig.update_layout(
        title=f"{ticker.upper()} RSI",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=360,
        margin=dict(l=30, r=30, t=60, b=40),
    )

    return fig


def make_macd_chart(price_df: pd.DataFrame, ticker: str) -> go.Figure:
    df = add_all_indicators(price_df).copy()

    fig = go.Figure()

    if "MACD" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                mode="lines",
                name="MACD",
            )
        )

    if "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                mode="lines",
                name="MACD Signal",
            )
        )

    if "MACD_Hist" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="MACD Histogram",
            )
        )

    fig.update_layout(
        title=f"{ticker.upper()} MACD",
        xaxis_title="Date",
        yaxis_title="MACD",
        height=360,
        margin=dict(l=30, r=30, t=60, b=40),
    )

    return fig


def make_sentiment_table(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    if sentiment_df is None or sentiment_df.empty:
        return pd.DataFrame(
            columns=[
                "Text",
                "Label",
                "Score",
                "Confidence",
                "Positive",
                "Negative",
                "Neutral",
            ]
        )

    df = sentiment_df.copy()

    display = pd.DataFrame(
        {
            "Text": df["text"].apply(lambda x: clean_text(x)[:180]),
            "Label": df["label"],
            "Score": df["sentiment_score"].round(4),
            "Confidence": (df["confidence"] * 100).round(2).astype(str) + "%",
            "Positive": (df["positive_prob"] * 100).round(2).astype(str) + "%",
            "Negative": (df["negative_prob"] * 100).round(2).astype(str) + "%",
            "Neutral": (df["neutral_prob"] * 100).round(2).astype(str) + "%",
        }
    )

    return display


def get_manual_news_items(text: str) -> List[str]:
    items = split_user_news_input(text)

    if not items:
        cleaned = clean_text(text)
        if cleaned:
            items = [cleaned]

    return items


def load_news_items(
    ticker: str,
    company_name: str,
    manual_news: str,
    use_live_news: bool,
    max_news_items: int,
) -> tuple[List[str], pd.DataFrame]:
    manual_items = get_manual_news_items(manual_news)

    news_df = pd.DataFrame()
    live_items = []

    if use_live_news:
        try:
            news_df = fetch_all_news(
                ticker=ticker,
                company_name=company_name,
                max_items=max_news_items,
            )
            live_items = news_dataframe_to_text_list(news_df)

        except Exception as error:
            st.warning(f"Live news fetch failed: {error}")

    combined_items = manual_items + live_items

    cleaned_items = []

    for item in combined_items:
        item = clean_text(item)
        if item and item not in cleaned_items:
            cleaned_items.append(item)

    return cleaned_items[:max_news_items], news_df


def show_metric_cards(signal_result: Dict[str, Any]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Final Signal", signal_result.get("final_signal", "HOLD"))
    col2.metric("Confidence", signal_result.get("confidence_label", "Low"))
    col3.metric("Final Score", round(safe_float(signal_result.get("final_score")), 4))
    col4.metric("Risk Level", signal_result.get("risk_level", "Low"))
    col5.metric("Sentiment", round(safe_float(signal_result.get("sentiment_score")), 4))


def show_price_summary(price_df: pd.DataFrame) -> None:
    if price_df is None or price_df.empty:
        return

    df = price_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(df["Volume"], errors="coerce").dropna() if "Volume" in df.columns else pd.Series(dtype=float)

    if close.empty:
        return

    last_close = float(close.iloc[-1])

    if len(close) >= 2:
        prev_close = float(close.iloc[-2])
        change = last_close - prev_close
        change_pct = change / prev_close if prev_close else 0.0
    else:
        change = 0.0
        change_pct = 0.0

    latest_volume = float(volume.iloc[-1]) if not volume.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Last Close", f"{last_close:.2f}")
    col2.metric("Change", f"{change:.2f}")
    col3.metric("Change %", f"{change_pct * 100:.2f}%")
    col4.metric("Volume", format_large_number(latest_volume))
    col5.metric("Market Cap", "N/A")


def main() -> None:
    st.title(APP_NAME)
    st.caption(APP_DESCRIPTION)

    with st.sidebar:
        st.header("Inputs")

        ticker = st.text_input("Ticker", value=DEFAULT_TICKER)
        company_name = st.text_input("Company Name", value=DEFAULT_COMPANY_NAME)

        period = st.selectbox(
            "Price Period",
            options=SUPPORTED_PERIODS,
            index=SUPPORTED_PERIODS.index(DEFAULT_PERIOD)
            if DEFAULT_PERIOD in SUPPORTED_PERIODS
            else 2,
        )

        interval = st.selectbox(
            "Price Interval",
            options=SUPPORTED_INTERVALS,
            index=SUPPORTED_INTERVALS.index(DEFAULT_INTERVAL)
            if DEFAULT_INTERVAL in SUPPORTED_INTERVALS
            else 0,
        )

        use_live_news = st.checkbox("Fetch live news", value=True)
        use_ml_model = st.checkbox("Use trained ML signal model if available", value=True)
        show_fibonacci = st.checkbox("Show Fibonacci levels on chart", value=True)

        max_news_items = st.slider("Max news items", min_value=1, max_value=25, value=10)

        run_button = st.button("Analyze", type="primary", use_container_width=True)

    default_news = """Apple reports stronger than expected earnings and raises guidance.
Analysts remain positive on Apple but warn about slowing hardware demand.
Apple announces new AI features expected to improve user engagement."""

    manual_news = st.text_area(
        "Manual news/headlines",
        value=default_news,
        height=160,
        help="Use one news item per line.",
    )

    st.markdown(DISCLAIMER)

    if not run_button:
        st.info("Enter a ticker and news, then click Analyze.")
        return

    ticker = ticker_clean(ticker)

    if not ticker:
        st.error("Ticker cannot be empty.")
        return

    try:
        with st.spinner("Fetching price data..."):
            price_df = get_price_data(
                ticker=ticker,
                period=period,
                interval=interval,
            )

        if price_df is None or price_df.empty:
            st.error("No price data found.")
            return

        with st.spinner("Fetching/preparing news..."):
            news_items, news_df = load_news_items(
                ticker=ticker,
                company_name=company_name,
                manual_news=manual_news,
                use_live_news=use_live_news,
                max_news_items=max_news_items,
            )

        if not news_items:
            st.warning("No valid news items found. Signal will use neutral sentiment.")

        with st.spinner("Running fine-tuned FinBERT sentiment..."):
            sentiment_df = analyze_news_batch(news_items)
            sentiment_summary = aggregate_sentiment(sentiment_df)

        with st.spinner("Generating Buy/Hold/Sell signal..."):
            signal_result = generate_signal(
                price_df=price_df,
                sentiment_summary=sentiment_summary,
                sentiment_df=sentiment_df,
                ticker=ticker,
                use_ml_model=use_ml_model,
            )

        st.subheader("Final Decision")
        show_metric_cards(signal_result)
        st.info(signal_result.get("message", ""))

        try:
            show_price_summary(price_df)
        except Exception as error:
            st.warning(f"Price summary unavailable: {error}")

        overview_tab, charts_tab, sentiment_tab, signal_tab, advanced_tab, risk_tab, ml_tab, news_tab = st.tabs(
            [
                "Overview",
                "Charts",
                "Sentiment",
                "Signal Breakdown",
                "Advanced Indicators",
                "Risk Filters",
                "ML Signal",
                "News",
            ]
        )

        with overview_tab:
            st.subheader("Signal Summary")
            st.dataframe(
                build_signal_table(signal_result),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Feature Contributions")
            st.dataframe(
                build_feature_contribution_table(signal_result),
                use_container_width=True,
                hide_index=True,
            )

        with charts_tab:
            st.plotly_chart(
                make_price_chart(price_df, ticker, show_fibonacci=show_fibonacci),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)

            with c1:
                st.plotly_chart(make_rsi_chart(price_df, ticker), use_container_width=True)

            with c2:
                st.plotly_chart(make_macd_chart(price_df, ticker), use_container_width=True)

            st.subheader("Fibonacci Levels")
            try:
                st.dataframe(
                    build_fibonacci_table(price_df),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as error:
                st.warning(f"Could not build Fibonacci table: {error}")

        with sentiment_tab:
            st.subheader("Sentiment Summary")

            summary_rows = [
                {"Metric": "Sentiment Score", "Value": sentiment_summary.get("sentiment_score", 0.0)},
                {"Metric": "Average Confidence", "Value": sentiment_summary.get("average_confidence", 0.0)},
                {"Metric": "News Count", "Value": sentiment_summary.get("news_count", 0)},
                {"Metric": "Positive Count", "Value": sentiment_summary.get("positive_count", 0)},
                {"Metric": "Negative Count", "Value": sentiment_summary.get("negative_count", 0)},
                {"Metric": "Neutral Count", "Value": sentiment_summary.get("neutral_count", 0)},
                {"Metric": "Dominant Label", "Value": sentiment_summary.get("dominant_label", "neutral")},
            ]

            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.subheader("Per-news FinBERT Results")
            st.dataframe(
                make_sentiment_table(sentiment_df),
                use_container_width=True,
                hide_index=True,
            )

        with signal_tab:
            st.subheader("Raw Signal Data")
            st.json(
                {
                    "final_signal": signal_result.get("final_signal"),
                    "raw_signal": signal_result.get("signal"),
                    "final_score": signal_result.get("final_score"),
                    "preliminary_score": signal_result.get("preliminary_score"),
                    "confidence": signal_result.get("confidence"),
                    "risk_level": signal_result.get("risk_level"),
                    "override": signal_result.get("override"),
                }
            )

            st.subheader("Indicator State")
            st.json(signal_result.get("indicator_summary", {}))

        with advanced_tab:
            advanced_signal = signal_result.get("advanced_signal", {})

            st.subheader("Advanced Indicator Score")
            st.dataframe(
                build_advanced_signal_table(advanced_signal),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Advanced Indicator Reasons")
            st.dataframe(
                build_advanced_reason_table(advanced_signal),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Advanced indicator raw values"):
                st.json(advanced_signal.get("advanced_indicators", {}))

        with risk_tab:
            risk_result = signal_result.get("risk_result", {})

            st.subheader("Risk Result")
            risk_cols = st.columns(4)
            risk_cols[0].metric("Risk Level", risk_result.get("risk_level", "Low"))
            risk_cols[1].metric("Risk Penalty", risk_result.get("risk_penalty", 0.0))
            risk_cols[2].metric("Risk Score", risk_result.get("risk_score", 0.0))
            risk_cols[3].metric("Blocker", str(risk_result.get("has_blocker", False)))

            st.subheader("Triggered Risk Filters")
            st.dataframe(
                build_risk_table(risk_result),
                use_container_width=True,
                hide_index=True,
            )

        with ml_tab:
            ml_result = signal_result.get("ml_signal", {})

            st.subheader("ML Signal Model")
            st.dataframe(
                build_ml_signal_table(ml_result),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("ML raw output"):
                st.json(ml_result)

        with news_tab:
            st.subheader("News Used")
            st.write(f"Total news/headlines used: {len(news_items)}")

            for idx, item in enumerate(news_items, start=1):
                st.markdown(f"**{idx}.** {item}")

            if news_df is not None and not news_df.empty:
                st.subheader("Fetched News Table")
                try:
                    st.dataframe(
                        build_news_display_table(news_df),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception:
                    st.dataframe(news_df, use_container_width=True)

        st.warning(DISCLAIMER)

    except Exception as error:
        st.error(f"Analysis failed: {error}")

        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()