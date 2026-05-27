from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_BUY_LEVEL,
    BACKTEST_SELL_LEVEL,
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    TRANSACTION_COST,
)
from src.indicators import add_all_indicators, get_latest_indicators
from src.signal_engine import (
    calculate_final_score,
    calculate_momentum_score,
    calculate_risk_score,
    calculate_trend_score,
    calculate_volume_score,
    label_from_score,
)
from src.utils import clean_numeric_series, logger, safe_float


def _empty_backtest_result() -> Dict[str, Any]:
    return {
        "success": False,
        "trades": pd.DataFrame(),
        "equity_curve": pd.DataFrame(),
        "metrics": {},
        "message": "Backtest could not be completed.",
    }


def prepare_backtest_data(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = add_all_indicators(price_df)

    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["Close"] = clean_numeric_series(df["Close"])
    df["Return_Next"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["Return_Next"] = df["Return_Next"].replace([np.inf, -np.inf], np.nan)

    return df.dropna(subset=["Close"])


def generate_historical_signal_scores(
    price_df: pd.DataFrame,
    sentiment_score: float = 0.0,
) -> pd.DataFrame:
    df = prepare_backtest_data(price_df)

    if df.empty:
        return pd.DataFrame()

    rows = []

    for idx in range(len(df)):
        window_df = df.iloc[: idx + 1].copy()

        if len(window_df) < 30:
            rows.append(
                {
                    "date": df.index[idx],
                    "final_score": 0.0,
                    "signal": "HOLD",
                    "trend_score": 0.0,
                    "momentum_score": 0.0,
                    "volume_score": 0.0,
                    "risk_score": 0.5,
                }
            )
            continue

        indicators = get_latest_indicators(window_df)

        trend_score = calculate_trend_score(indicators)
        momentum_score = calculate_momentum_score(indicators)
        volume_score = calculate_volume_score(indicators)
        risk_score = calculate_risk_score(indicators, window_df)

        features = {
            "sentiment_score": safe_float(sentiment_score, 0.0),
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "volume_score": volume_score,
            "risk_score": risk_score,
        }

        final_score = calculate_final_score(features)
        signal = label_from_score(final_score)

        rows.append(
            {
                "date": df.index[idx],
                "final_score": final_score,
                "signal": signal,
                "trend_score": trend_score,
                "momentum_score": momentum_score,
                "volume_score": volume_score,
                "risk_score": risk_score,
            }
        )

    signal_df = pd.DataFrame(rows).set_index("date")
    result = df.join(signal_df, how="left")

    return result


def signal_to_position(final_score: float) -> int:
    final_score = safe_float(final_score, 0.0)

    if final_score >= BACKTEST_BUY_LEVEL:
        return 1

    if final_score <= BACKTEST_SELL_LEVEL:
        return -1

    return 0


def run_backtest(
    price_df: pd.DataFrame,
    sentiment_score: float = 0.0,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
    allow_short: bool = False,
) -> Dict[str, Any]:
    try:
        df = generate_historical_signal_scores(
            price_df=price_df,
            sentiment_score=sentiment_score,
        )

        if df.empty:
            result = _empty_backtest_result()
            result["message"] = "No valid price data available for backtest."
            return result

        df = df.copy()

        df["Raw_Position"] = df["final_score"].apply(signal_to_position)

        if not allow_short:
            df["Raw_Position"] = df["Raw_Position"].clip(lower=0)

        df["Position"] = df["Raw_Position"].shift(1).fillna(0)

        df["Asset_Return"] = df["Close"].pct_change().fillna(0.0)
        df["Strategy_Return_Before_Cost"] = df["Position"] * df["Asset_Return"]

        df["Trade"] = df["Position"].diff().abs().fillna(df["Position"].abs())
        df["Cost"] = df["Trade"] * transaction_cost

        df["Strategy_Return"] = df["Strategy_Return_Before_Cost"] - df["Cost"]
        df["Benchmark_Return"] = df["Asset_Return"]

        df["Strategy_Equity"] = initial_capital * (1.0 + df["Strategy_Return"]).cumprod()
        df["Benchmark_Equity"] = initial_capital * (1.0 + df["Benchmark_Return"]).cumprod()

        df["Strategy_Peak"] = df["Strategy_Equity"].cummax()
        df["Drawdown"] = df["Strategy_Equity"] / df["Strategy_Peak"] - 1.0

        trades = extract_trades(df)
        metrics = calculate_backtest_metrics(df, trades, initial_capital)

        equity_curve = df[
            [
                "Close",
                "final_score",
                "signal",
                "Position",
                "Strategy_Return",
                "Benchmark_Return",
                "Strategy_Equity",
                "Benchmark_Equity",
                "Drawdown",
            ]
        ].copy()

        return {
            "success": True,
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "message": "Backtest completed successfully.",
        }

    except Exception as error:
        logger.exception("Backtest failed: %s", error)
        result = _empty_backtest_result()
        result["message"] = str(error)
        return result


def extract_trades(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if backtest_df is None or backtest_df.empty:
        return pd.DataFrame()

    df = backtest_df.copy()

    if "Position" not in df.columns:
        return pd.DataFrame()

    trades = []
    current_trade = None

    for date, row in df.iterrows():
        position = int(safe_float(row.get("Position"), 0))
        close = safe_float(row.get("Close"), 0.0)

        if current_trade is None and position != 0:
            current_trade = {
                "entry_date": date,
                "entry_price": close,
                "position": position,
            }

        elif current_trade is not None and position == 0:
            entry_price = safe_float(current_trade["entry_price"], close)
            trade_position = int(current_trade["position"])

            if entry_price != 0:
                if trade_position > 0:
                    trade_return = close / entry_price - 1.0
                else:
                    trade_return = entry_price / close - 1.0
            else:
                trade_return = 0.0

            current_trade.update(
                {
                    "exit_date": date,
                    "exit_price": close,
                    "return": trade_return,
                    "holding_days": max(1, len(df.loc[current_trade["entry_date"] : date])),
                }
            )

            trades.append(current_trade)
            current_trade = None

    if current_trade is not None:
        final_date = df.index[-1]
        final_close = safe_float(df["Close"].iloc[-1], current_trade["entry_price"])
        entry_price = safe_float(current_trade["entry_price"], final_close)
        trade_position = int(current_trade["position"])

        if entry_price != 0:
            if trade_position > 0:
                trade_return = final_close / entry_price - 1.0
            else:
                trade_return = entry_price / final_close - 1.0
        else:
            trade_return = 0.0

        current_trade.update(
            {
                "exit_date": final_date,
                "exit_price": final_close,
                "return": trade_return,
                "holding_days": max(1, len(df.loc[current_trade["entry_date"] : final_date])),
            }
        )

        trades.append(current_trade)

    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "position",
                "return",
                "holding_days",
            ]
        )

    trades_df = pd.DataFrame(trades)

    return trades_df[
        [
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "position",
            "return",
            "holding_days",
        ]
    ]


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    returns = clean_numeric_series(returns).dropna()

    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf

    std = safe_float(excess_returns.std(), 0.0)

    if std == 0:
        return 0.0

    sharpe = np.sqrt(252) * safe_float(excess_returns.mean(), 0.0) / std

    return round(float(sharpe), 4)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    returns = clean_numeric_series(returns).dropna()

    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / 252.0
    excess_returns = returns - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    downside_std = safe_float(downside_returns.std(), 0.0)

    if downside_std == 0:
        return 0.0

    sortino = np.sqrt(252) * safe_float(excess_returns.mean(), 0.0) / downside_std

    return round(float(sortino), 4)


def calculate_max_drawdown(equity: pd.Series) -> float:
    equity = clean_numeric_series(equity).dropna()

    if equity.empty:
        return 0.0

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    return round(float(drawdown.min()), 6)


def calculate_backtest_metrics(
    backtest_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> Dict[str, Any]:
    if backtest_df is None or backtest_df.empty:
        return {}

    df = backtest_df.copy()

    strategy_equity = clean_numeric_series(df["Strategy_Equity"]).dropna()
    benchmark_equity = clean_numeric_series(df["Benchmark_Equity"]).dropna()

    if strategy_equity.empty:
        return {}

    final_equity = safe_float(strategy_equity.iloc[-1], initial_capital)
    benchmark_final_equity = safe_float(benchmark_equity.iloc[-1], initial_capital)

    total_return = final_equity / initial_capital - 1.0
    benchmark_return = benchmark_final_equity / initial_capital - 1.0

    strategy_returns = clean_numeric_series(df["Strategy_Return"]).fillna(0.0)

    days = max(1, len(df))
    years = days / 252.0

    if years > 0:
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0
        benchmark_annualized_return = (1.0 + benchmark_return) ** (1.0 / years) - 1.0
    else:
        annualized_return = 0.0
        benchmark_annualized_return = 0.0

    max_drawdown = calculate_max_drawdown(strategy_equity)
    sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
    sortino_ratio = calculate_sortino_ratio(strategy_returns)

    number_of_trades = 0
    win_rate = 0.0
    average_trade_return = 0.0
    best_trade = 0.0
    worst_trade = 0.0
    profit_factor = 0.0
    average_holding_days = 0.0

    if trades_df is not None and not trades_df.empty and "return" in trades_df.columns:
        trade_returns = clean_numeric_series(trades_df["return"]).dropna()
        number_of_trades = int(len(trade_returns))

        if number_of_trades > 0:
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns < 0]

            win_rate = len(wins) / number_of_trades
            average_trade_return = safe_float(trade_returns.mean(), 0.0)
            best_trade = safe_float(trade_returns.max(), 0.0)
            worst_trade = safe_float(trade_returns.min(), 0.0)

            gross_profit = safe_float(wins.sum(), 0.0)
            gross_loss = abs(safe_float(losses.sum(), 0.0))

            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

            if "holding_days" in trades_df.columns:
                average_holding_days = safe_float(trades_df["holding_days"].mean(), 0.0)

    active_days = int((df.get("Position", pd.Series(index=df.index, data=0)) != 0).sum())
    exposure = active_days / len(df) if len(df) else 0.0

    turnover = safe_float(df.get("Trade", pd.Series(index=df.index, data=0)).sum(), 0.0)

    return {
        "initial_capital": round(float(initial_capital), 2),
        "final_equity": round(float(final_equity), 2),
        "benchmark_final_equity": round(float(benchmark_final_equity), 2),
        "total_return": round(float(total_return), 6),
        "benchmark_return": round(float(benchmark_return), 6),
        "excess_return": round(float(total_return - benchmark_return), 6),
        "annualized_return": round(float(annualized_return), 6),
        "benchmark_annualized_return": round(float(benchmark_annualized_return), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "number_of_trades": number_of_trades,
        "win_rate": round(float(win_rate), 6),
        "average_trade_return": round(float(average_trade_return), 6),
        "best_trade": round(float(best_trade), 6),
        "worst_trade": round(float(worst_trade), 6),
        "profit_factor": round(float(profit_factor), 6) if np.isfinite(profit_factor) else "Infinity",
        "average_holding_days": round(float(average_holding_days), 2),
        "exposure": round(float(exposure), 6),
        "turnover": round(float(turnover), 4),
        "data_points": int(len(df)),
    }


def build_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(columns=["Metric", "Value"])

    display_names = {
        "initial_capital": "Initial Capital",
        "final_equity": "Final Strategy Equity",
        "benchmark_final_equity": "Final Benchmark Equity",
        "total_return": "Strategy Total Return",
        "benchmark_return": "Benchmark Return",
        "excess_return": "Excess Return",
        "annualized_return": "Annualized Return",
        "benchmark_annualized_return": "Benchmark Annualized Return",
        "max_drawdown": "Max Drawdown",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "number_of_trades": "Number of Trades",
        "win_rate": "Win Rate",
        "average_trade_return": "Average Trade Return",
        "best_trade": "Best Trade",
        "worst_trade": "Worst Trade",
        "profit_factor": "Profit Factor",
        "average_holding_days": "Average Holding Days",
        "exposure": "Market Exposure",
        "turnover": "Turnover",
        "data_points": "Data Points",
    }

    percentage_keys = {
        "total_return",
        "benchmark_return",
        "excess_return",
        "annualized_return",
        "benchmark_annualized_return",
        "max_drawdown",
        "win_rate",
        "average_trade_return",
        "best_trade",
        "worst_trade",
        "exposure",
    }

    rows = []

    for key, label in display_names.items():
        if key not in metrics:
            continue

        value = metrics[key]

        if key in percentage_keys and isinstance(value, (int, float)):
            formatted_value = f"{value * 100:.2f}%"
        elif isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        rows.append(
            {
                "Metric": label,
                "Value": formatted_value,
            }
        )

    return pd.DataFrame(rows)


def build_equity_chart_data(backtest_result: Dict[str, Any]) -> pd.DataFrame:
    if not backtest_result or not backtest_result.get("success"):
        return pd.DataFrame()

    equity_curve = backtest_result.get("equity_curve")

    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame()

    df = equity_curve.copy().reset_index()

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    required_cols = ["Date", "Strategy_Equity", "Benchmark_Equity"]

    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    return df[required_cols]