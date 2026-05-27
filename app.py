import traceback
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.config import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_COMPANY_NAME,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_TICKER,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    HF_SPACE_MODE,
    SUPPORTED_INTERVALS,
    SUPPORTED_PERIODS,
)
from src.indicators import add_all_indicators
from src.market_data import get_price_data, get_ticker_info, get_latest_price_summary
from src.scraper import (
    build_news_display_table,
    fetch_all_news,
    news_dataframe_to_text_list,
)
from src.sentiment import aggregate_sentiment, analyze_news_batch
from src.signal_engine import build_signal_table, generate_signal
from src.utils import clean_text, logger, split_user_news_input, ticker_clean


DISCLAIMER = """
FinSentinel is an educational signal analysis tool. It is not financial advice.
Always verify information independently before making investment decisions.
"""


def make_empty_price_chart() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Price chart unavailable",
        xaxis_title="Date",
        yaxis_title="Price",
        height=420,
        margin=dict(l=40, r=30, t=60, b=40),
    )
    return fig


def make_price_chart(price_df: pd.DataFrame, ticker: str) -> go.Figure:
    if price_df is None or price_df.empty:
        return make_empty_price_chart()

    df = add_all_indicators(price_df).copy()

    if df.empty or "Close" not in df.columns:
        return make_empty_price_chart()

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

    fig.update_layout(
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def make_indicator_chart(price_df: pd.DataFrame, ticker: str) -> go.Figure:
    if price_df is None or price_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Indicator chart unavailable",
            height=420,
            margin=dict(l=40, r=30, t=60, b=40),
        )
        return fig

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
        title=f"{ticker} RSI",
        xaxis_title="Date",
        yaxis_title="RSI",
        height=380,
        margin=dict(l=40, r=30, t=60, b=40),
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
            "Score": df["sentiment_score"].round(3),
            "Confidence": (df["confidence"] * 100).round(2).astype(str) + "%",
            "Positive": (df["positive_prob"] * 100).round(2).astype(str) + "%",
            "Negative": (df["negative_prob"] * 100).round(2).astype(str) + "%",
            "Neutral": (df["neutral_prob"] * 100).round(2).astype(str) + "%",
        }
    )

    return display


def build_status_markdown(
    ticker: str,
    company_name: str,
    signal_result: Dict[str, Any],
    price_summary: Dict[str, Any],
    ticker_info: Dict[str, Any],
) -> str:
    signal = signal_result.get("signal", "HOLD")
    confidence = signal_result.get("confidence", 0.0)
    risk_label = signal_result.get("risk_label", "Medium")
    final_score = signal_result.get("final_score", 0.0)

    latest_close = price_summary.get("latest_close", 0.0)
    daily_return = price_summary.get("daily_return", 0.0)
    currency = ticker_info.get("currency", "")

    return f"""
## {signal}

**Ticker:** {ticker}  
**Company:** {company_name or ticker_info.get("long_name", ticker)}  
**Confidence:** {confidence:.2f}%  
**Risk Level:** {risk_label}  
**Final Score:** {final_score:.3f}  

**Latest Close:** {latest_close} {currency}  
**Daily Return:** {daily_return * 100:.2f}%  

{DISCLAIMER}
"""


def analyze_manual_news(
    ticker: str,
    company_name: str,
    manual_news: str,
    period: str,
    interval: str,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, str]:
    ticker = ticker_clean(ticker)

    if not ticker:
        raise gr.Error("Please enter a valid ticker.")

    news_items = split_user_news_input(manual_news)

    if not news_items:
        raise gr.Error("Please enter at least one news headline or text.")

    price_df = get_price_data(ticker=ticker, period=period, interval=interval)
    ticker_info = get_ticker_info(ticker)
    price_summary = get_latest_price_summary(price_df)

    sentiment_df = analyze_news_batch(news_items)
    sentiment_summary = aggregate_sentiment(sentiment_df)

    signal_result = generate_signal(
        sentiment_summary=sentiment_summary,
        price_df=price_df,
    )

    status_md = build_status_markdown(
        ticker=ticker,
        company_name=company_name,
        signal_result=signal_result,
        price_summary=price_summary,
        ticker_info=ticker_info,
    )

    sentiment_table = make_sentiment_table(sentiment_df)
    signal_table = build_signal_table(signal_result)
    price_chart = make_price_chart(price_df, ticker)
    indicator_chart = make_indicator_chart(price_df, ticker)
    explanation = signal_result.get("explanation", "No explanation available.")

    return (
        status_md,
        signal_table,
        sentiment_table,
        price_chart,
        indicator_chart,
        explanation,
    )


def analyze_live_news(
    ticker: str,
    company_name: str,
    period: str,
    interval: str,
    max_news: int,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, go.Figure, go.Figure, str]:
    ticker = ticker_clean(ticker)

    if not ticker:
        raise gr.Error("Please enter a valid ticker.")

    price_df = get_price_data(ticker=ticker, period=period, interval=interval)
    ticker_info = get_ticker_info(ticker)
    price_summary = get_latest_price_summary(price_df)

    news_df = fetch_all_news(
        ticker=ticker,
        company_name=company_name,
        max_items=int(max_news),
    )

    news_items = news_dataframe_to_text_list(news_df)

    if not news_items:
        sentiment_df = analyze_news_batch([])
    else:
        sentiment_df = analyze_news_batch(news_items)

    sentiment_summary = aggregate_sentiment(sentiment_df)

    signal_result = generate_signal(
        sentiment_summary=sentiment_summary,
        price_df=price_df,
    )

    status_md = build_status_markdown(
        ticker=ticker,
        company_name=company_name,
        signal_result=signal_result,
        price_summary=price_summary,
        ticker_info=ticker_info,
    )

    news_table = build_news_display_table(news_df)
    sentiment_table = make_sentiment_table(sentiment_df)
    signal_table = build_signal_table(signal_result)
    price_chart = make_price_chart(price_df, ticker)
    indicator_chart = make_indicator_chart(price_df, ticker)
    explanation = signal_result.get("explanation", "No explanation available.")

    return (
        status_md,
        signal_table,
        news_table,
        sentiment_table,
        price_chart,
        indicator_chart,
        explanation,
    )


def safe_manual_wrapper(
    ticker: str,
    company_name: str,
    manual_news: str,
    period: str,
    interval: str,
):
    try:
        return analyze_manual_news(
            ticker=ticker,
            company_name=company_name,
            manual_news=manual_news,
            period=period,
            interval=interval,
        )

    except gr.Error:
        raise

    except Exception as error:
        logger.exception("Manual analysis failed: %s", error)
        error_message = f"Analysis failed: {error}"
        traceback_text = traceback.format_exc(limit=2)

        return (
            f"## ERROR\n\n{error_message}",
            pd.DataFrame(columns=["Metric", "Value"]),
            pd.DataFrame(),
            make_empty_price_chart(),
            make_empty_price_chart(),
            traceback_text,
        )


def safe_live_wrapper(
    ticker: str,
    company_name: str,
    period: str,
    interval: str,
    max_news: int,
):
    try:
        return analyze_live_news(
            ticker=ticker,
            company_name=company_name,
            period=period,
            interval=interval,
            max_news=max_news,
        )

    except gr.Error:
        raise

    except Exception as error:
        logger.exception("Live news analysis failed: %s", error)
        error_message = f"Analysis failed: {error}"
        traceback_text = traceback.format_exc(limit=2)

        return (
            f"## ERROR\n\n{error_message}",
            pd.DataFrame(columns=["Metric", "Value"]),
            pd.DataFrame(),
            pd.DataFrame(),
            make_empty_price_chart(),
            make_empty_price_chart(),
            traceback_text,
        )


custom_css = """
#main-title {
    text-align: center;
}
.signal-box {
    border-radius: 16px;
}
"""


with gr.Blocks(
    title=APP_NAME,
    css=custom_css,
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(f"# {APP_NAME}", elem_id="main-title")
    gr.Markdown(APP_DESCRIPTION)
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        ticker_input = gr.Textbox(
            label="Ticker",
            value=DEFAULT_TICKER,
            placeholder="Example: AAPL, MSFT, TSLA, INFY.NS",
        )

        company_input = gr.Textbox(
            label="Company Name",
            value=DEFAULT_COMPANY_NAME,
            placeholder="Example: Apple",
        )

    with gr.Row():
        period_input = gr.Dropdown(
            label="Price Period",
            choices=SUPPORTED_PERIODS,
            value=DEFAULT_PERIOD,
        )

        interval_input = gr.Dropdown(
            label="Price Interval",
            choices=SUPPORTED_INTERVALS,
            value=DEFAULT_INTERVAL,
        )

    with gr.Tab("Manual News Analysis"):
        manual_news_input = gr.Textbox(
            label="Paste news headlines or financial text",
            lines=7,
            placeholder=(
                "Paste one headline per line.\n"
                "Example: Apple reports stronger-than-expected quarterly earnings."
            ),
        )

        manual_button = gr.Button("Analyze Manual News", variant="primary")

        manual_status = gr.Markdown()
        manual_signal_table = gr.Dataframe(label="Signal Breakdown")
        manual_sentiment_table = gr.Dataframe(label="Sentiment Table")
        manual_price_chart = gr.Plot(label="Price Chart")
        manual_indicator_chart = gr.Plot(label="Indicator Chart")
        manual_explanation = gr.Textbox(
            label="Explanation",
            lines=6,
        )

        manual_button.click(
            fn=safe_manual_wrapper,
            inputs=[
                ticker_input,
                company_input,
                manual_news_input,
                period_input,
                interval_input,
            ],
            outputs=[
                manual_status,
                manual_signal_table,
                manual_sentiment_table,
                manual_price_chart,
                manual_indicator_chart,
                manual_explanation,
            ],
        )

    with gr.Tab("Live News Analysis"):
        max_news_input = gr.Slider(
            label="Maximum News Items",
            minimum=3,
            maximum=25,
            value=10,
            step=1,
        )

        live_button = gr.Button("Fetch Live News and Analyze", variant="primary")

        live_status = gr.Markdown()
        live_signal_table = gr.Dataframe(label="Signal Breakdown")
        live_news_table = gr.Dataframe(label="Fetched News")
        live_sentiment_table = gr.Dataframe(label="Sentiment Table")
        live_price_chart = gr.Plot(label="Price Chart")
        live_indicator_chart = gr.Plot(label="Indicator Chart")
        live_explanation = gr.Textbox(
            label="Explanation",
            lines=6,
        )

        live_button.click(
            fn=safe_live_wrapper,
            inputs=[
                ticker_input,
                company_input,
                period_input,
                interval_input,
                max_news_input,
            ],
            outputs=[
                live_status,
                live_signal_table,
                live_news_table,
                live_sentiment_table,
                live_price_chart,
                live_indicator_chart,
                live_explanation,
            ],
        )

    with gr.Accordion("How FinSentinel works", open=False):
        gr.Markdown(
            """
FinSentinel combines financial news sentiment, market price data, technical indicators, volume confirmation, and risk filters.

The final signal is not directly equal to FinBERT sentiment. FinBERT produces a sentiment score, and the signal engine combines it with trend, momentum, volume, and risk scores.

Main outputs:
- BUY / HOLD / SELL signal
- confidence score
- risk level
- explanation
- sentiment table
- technical chart
"""
        )


if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_SERVER_NAME if HF_SPACE_MODE else None,
        server_port=GRADIO_SERVER_PORT if HF_SPACE_MODE else None,
        share=True,
        inbrowser=True,
        show_error=True,
    )