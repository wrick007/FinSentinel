import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.market_data import get_price_data
from src.signal_engine import generate_signal
from src.utils import safe_float


def make_synthetic_sentiment(row: pd.Series) -> Dict[str, Any]:
    momentum_5d = safe_float(row.get("Return_5D", row.get("return_5d", 0.0)), 0.0)
    momentum_20d = safe_float(row.get("Return_20D", row.get("return_20d", 0.0)), 0.0)

    raw_score = (0.60 * momentum_5d) + (0.40 * momentum_20d)
    sentiment_score = max(-0.70, min(0.70, raw_score * 6.0))

    if sentiment_score > 0.15:
        dominant_label = "positive"
        positive_count = 3
        negative_count = 1
        neutral_count = 1
    elif sentiment_score < -0.15:
        dominant_label = "negative"
        positive_count = 1
        negative_count = 3
        neutral_count = 1
    else:
        dominant_label = "neutral"
        positive_count = 1
        negative_count = 1
        neutral_count = 3

    confidence = min(0.95, 0.55 + abs(sentiment_score))

    return {
        "sentiment_score": round(float(sentiment_score), 6),
        "average_confidence": round(float(confidence), 6),
        "news_count": 5,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_ratio": positive_count / 5,
        "negative_ratio": negative_count / 5,
        "neutral_ratio": neutral_count / 5,
        "dominant_label": dominant_label,
    }


def build_backtest_windows(price_df: pd.DataFrame, min_window: int = 80):
    for idx in range(min_window, len(price_df) - 5):
        window_df = price_df.iloc[: idx + 1].copy()
        current_date = price_df.index[idx]
        current_row = price_df.iloc[idx]
        future_close = safe_float(price_df["Close"].iloc[idx + 5], np.nan)
        current_close = safe_float(price_df["Close"].iloc[idx], np.nan)

        if current_close <= 0 or np.isnan(future_close):
            continue

        future_return_5d = future_close / current_close - 1.0

        yield current_date, current_row, window_df, future_return_5d


def signal_to_position(signal: str, allow_short: bool = False) -> int:
    signal = str(signal).upper().strip()

    if signal in {"BUY", "STRONG BUY"}:
        return 1

    if signal in {"SELL", "STRONG SELL"}:
        return -1 if allow_short else 0

    return 0


def calculate_backtest_metrics(result_df: pd.DataFrame) -> Dict[str, Any]:
    if result_df.empty:
        return {
            "rows": 0,
            "trades": 0,
            "hit_rate": 0.0,
            "avg_strategy_return_5d": 0.0,
            "avg_forward_return_5d": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "hold_count": 0,
        }

    trades = result_df[result_df["position"] != 0].copy()

    if trades.empty:
        hit_rate = 0.0
        avg_strategy_return = 0.0
    else:
        hit_rate = float((trades["strategy_return_5d"] > 0).mean())
        avg_strategy_return = float(trades["strategy_return_5d"].mean())

    return {
        "rows": int(len(result_df)),
        "trades": int(len(trades)),
        "hit_rate": round(hit_rate, 6),
        "avg_strategy_return_5d": round(avg_strategy_return, 6),
        "avg_forward_return_5d": round(float(result_df["future_return_5d"].mean()), 6),
        "buy_count": int(result_df["final_signal"].isin(["BUY", "STRONG BUY"]).sum()),
        "sell_count": int(result_df["final_signal"].isin(["SELL", "STRONG SELL"]).sum()),
        "hold_count": int((result_df["final_signal"] == "HOLD").sum()),
        "avg_confidence": round(float(result_df["confidence"].mean()), 6),
        "avg_final_score": round(float(result_df["final_score"].mean()), 6),
    }


def run_signal_backtest(
    ticker: str = "AAPL",
    period: str = "2y",
    interval: str = "1d",
    allow_short: bool = False,
    use_ml_model: bool = False,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    price_df = get_price_data(
        ticker=ticker,
        period=period,
        interval=interval,
    )

    if price_df is None or price_df.empty:
        raise ValueError(f"No price data found for {ticker}.")

    if len(price_df) < 100:
        raise ValueError("Not enough price history for backtest. Use period='2y' or more.")

    price_df = price_df.copy()

    price_df["Return_5D"] = price_df["Close"].pct_change(5)
    price_df["Return_20D"] = price_df["Close"].pct_change(20)

    records = []

    for current_date, row, window_df, future_return_5d in build_backtest_windows(price_df):
        sentiment_summary = make_synthetic_sentiment(row)

        signal_result = generate_signal(
            price_df=window_df,
            sentiment_summary=sentiment_summary,
            sentiment_df=None,
            ticker=ticker,
            use_ml_model=use_ml_model,
        )

        final_signal = signal_result.get("final_signal", "HOLD")
        position = signal_to_position(final_signal, allow_short=allow_short)
        strategy_return_5d = position * future_return_5d

        records.append(
            {
                "date": current_date,
                "ticker": ticker,
                "close": safe_float(row.get("Close"), 0.0),
                "final_signal": final_signal,
                "raw_signal": signal_result.get("signal", "HOLD"),
                "position": position,
                "final_score": signal_result.get("final_score", 0.0),
                "confidence": signal_result.get("confidence", 0.0),
                "confidence_label": signal_result.get("confidence_label", "Low"),
                "sentiment_score": signal_result.get("sentiment_score", 0.0),
                "trend_score": signal_result.get("trend_score", 0.0),
                "momentum_score": signal_result.get("momentum_score", 0.0),
                "volume_score": signal_result.get("volume_score", 0.0),
                "advanced_indicator_score": signal_result.get("advanced_indicator_score", 0.0),
                "risk_score": signal_result.get("risk_score", 0.0),
                "risk_penalty": signal_result.get("risk_penalty", 0.0),
                "risk_level": signal_result.get("risk_level", "Low"),
                "future_return_5d": future_return_5d,
                "strategy_return_5d": strategy_return_5d,
                "message": signal_result.get("message", ""),
            }
        )

    result_df = pd.DataFrame(records)
    metrics = calculate_backtest_metrics(result_df)

    return result_df, metrics


def save_backtest_outputs(
    result_df: pd.DataFrame,
    metrics: Dict[str, Any],
    ticker: str,
) -> None:
    output_dir = PROJECT_ROOT / "data" / "processed" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace(".", "_").replace("-", "_")

    result_path = output_dir / f"{safe_ticker}_signal_backtest.csv"
    metrics_path = output_dir / f"{safe_ticker}_signal_backtest_metrics.json"

    result_df.to_csv(result_path, index=False)

    import json

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved backtest results to: {result_path}")
    print(f"Saved metrics to: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest FinSentinel signal engine.")

    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="2y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--use-ml-model", action="store_true")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    result_df, metrics = run_signal_backtest(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        allow_short=args.allow_short,
        use_ml_model=args.use_ml_model,
    )

    print("\nBacktest Metrics")
    print(metrics)

    print("\nRecent Signals")
    if not result_df.empty:
        print(result_df.tail(10).to_string(index=False))

    if args.save:
        save_backtest_outputs(
            result_df=result_df,
            metrics=metrics,
            ticker=args.ticker,
        )


if __name__ == "__main__":
    main()