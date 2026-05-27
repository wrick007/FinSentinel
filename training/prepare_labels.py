import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import PROCESSED_DATA_DIR
from src.utils import logger, safe_float, safe_str, ticker_clean


DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "feature_dataset.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "signal_dataset.csv"


LABEL_MAP = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2,
}

INVERSE_LABEL_MAP = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


def load_feature_dataset(input_path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {input_path}. "
            "Run training/build_dataset.py first."
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    required_columns = {"ticker", "date", "Close"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Feature dataset missing required columns: {sorted(missing)}")

    df["ticker"] = df["ticker"].apply(ticker_clean)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["ticker", "date", "Close"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def calculate_forward_returns(
    df: pd.DataFrame,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 3, 5, 10, 20]

    result = df.copy()

    for horizon in horizons:
        result[f"future_close_{horizon}d"] = (
            result.groupby("ticker")["Close"].shift(-horizon)
        )

        result[f"future_return_{horizon}d"] = (
            result[f"future_close_{horizon}d"] / result["Close"] - 1.0
        )

    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def create_three_class_label(
    future_return: Any,
    buy_threshold: float,
    sell_threshold: float,
) -> str:
    future_return = safe_float(future_return, np.nan)

    if np.isnan(future_return):
        return ""

    if future_return >= buy_threshold:
        return "BUY"

    if future_return <= sell_threshold:
        return "SELL"

    return "HOLD"


def create_binary_label(
    future_return: Any,
    threshold: float,
) -> int:
    future_return = safe_float(future_return, np.nan)

    if np.isnan(future_return):
        return -1

    return int(future_return >= threshold)


def add_signal_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    binary_threshold: float = 0.0,
) -> pd.DataFrame:
    result = df.copy()

    future_col = f"future_return_{horizon}d"

    if future_col not in result.columns:
        result = calculate_forward_returns(result, horizons=[horizon])

    result["label"] = result[future_col].apply(
        lambda x: create_three_class_label(
            future_return=x,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
    )

    result["label_id"] = result["label"].map(LABEL_MAP)
    result["binary_label"] = result[future_col].apply(
        lambda x: create_binary_label(
            future_return=x,
            threshold=binary_threshold,
        )
    )

    result["target_horizon"] = horizon
    result["buy_threshold"] = buy_threshold
    result["sell_threshold"] = sell_threshold

    result = result[result["label"].isin(LABEL_MAP.keys())].copy()
    result["label_id"] = result["label_id"].astype(int)

    return result


def add_risk_adjusted_targets(
    df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    result = df.copy()

    future_col = f"future_return_{horizon}d"

    if future_col not in result.columns:
        return result

    volatility_col = "Volatility_20D"

    if volatility_col not in result.columns:
        result["risk_adjusted_future_return"] = result[future_col]
        return result

    volatility = pd.to_numeric(result[volatility_col], errors="coerce")
    volatility = volatility.replace(0, np.nan)

    result["risk_adjusted_future_return"] = result[future_col] / volatility
    result["risk_adjusted_future_return"] = result[
        "risk_adjusted_future_return"
    ].replace([np.inf, -np.inf], np.nan)

    return result


def add_regression_targets(
    df: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    result = df.copy()

    future_col = f"future_return_{horizon}d"

    if future_col not in result.columns:
        result = calculate_forward_returns(result, horizons=[horizon])

    result["target_return"] = result[future_col]
    result["target_direction"] = (result[future_col] > 0).astype(int)

    return result


def remove_leaky_columns(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    result = df.copy()

    leaky_cols = []

    for col in result.columns:
        lower = col.lower()

        if lower.startswith("future_close_"):
            leaky_cols.append(col)

        if lower.startswith("future_return_") and col != f"future_return_{horizon}d":
            leaky_cols.append(col)

    if leaky_cols:
        result = result.drop(columns=leaky_cols, errors="ignore")

    return result


def clean_signal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    object_cols = result.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        result[col] = result[col].fillna("").astype(str)

    numeric_cols = [
        col
        for col in result.columns
        if col not in {"ticker", "date", "dominant_label", "label"}
    ]

    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.replace([np.inf, -np.inf], np.nan)

    essential_cols = [
        "ticker",
        "date",
        "Close",
        "target_return",
        "label",
        "label_id",
    ]

    result = result.dropna(subset=[col for col in essential_cols if col in result.columns])

    return result


def build_signal_dataset(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    horizon: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    binary_threshold: float = 0.0,
    drop_leaky_columns: bool = True,
) -> pd.DataFrame:
    logger.info("Loading feature dataset from %s", input_path)

    df = load_feature_dataset(input_path)

    logger.info("Feature dataset shape: %s", df.shape)

    df = calculate_forward_returns(
        df,
        horizons=sorted(set([1, 3, 5, 10, 20, horizon])),
    )

    df = add_signal_labels(
        df,
        horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        binary_threshold=binary_threshold,
    )

    df = add_regression_targets(df, horizon=horizon)
    df = add_risk_adjusted_targets(df, horizon=horizon)

    if drop_leaky_columns:
        df = remove_leaky_columns(df, horizon=horizon)

    df = clean_signal_dataset(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Saved signal dataset to %s", output_path)
    logger.info("Signal dataset shape: %s", df.shape)

    return df


def build_label_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "tickers": 0,
            "start_date": "",
            "end_date": "",
            "label_distribution": {},
            "avg_target_return": 0.0,
        }

    label_distribution = (
        df["label"].value_counts(normalize=False).to_dict()
        if "label" in df.columns
        else {}
    )

    label_distribution_pct = (
        (df["label"].value_counts(normalize=True) * 100).round(2).to_dict()
        if "label" in df.columns
        else {}
    )

    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "start_date": safe_str(df["date"].min()) if "date" in df.columns else "",
        "end_date": safe_str(df["date"].max()) if "date" in df.columns else "",
        "label_distribution": label_distribution,
        "label_distribution_pct": label_distribution_pct,
        "avg_target_return": round(
            safe_float(df["target_return"].mean(), 0.0),
            6,
        )
        if "target_return" in df.columns
        else 0.0,
        "median_target_return": round(
            safe_float(df["target_return"].median(), 0.0),
            6,
        )
        if "target_return" in df.columns
        else 0.0,
    }


def print_summary(summary: Dict[str, Any], output_path: Path) -> None:
    print("\nFinSentinel Label Preparation Summary")
    print("-------------------------------------")
    print(f"Rows: {summary['rows']}")
    print(f"Tickers: {summary['tickers']}")
    print(f"Start date: {summary['start_date']}")
    print(f"End date: {summary['end_date']}")
    print(f"Average target return: {summary['avg_target_return']}")
    print(f"Median target return: {summary['median_target_return']}")
    print("\nLabel distribution:")

    for label, count in summary["label_distribution"].items():
        pct = summary["label_distribution_pct"].get(label, 0.0)
        print(f"  {label}: {count} rows ({pct:.2f}%)")

    print(f"\nSaved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Buy/Hold/Sell labels from FinSentinel feature dataset."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input feature dataset CSV path.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output signal dataset CSV path.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward return horizon in trading days.",
    )

    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.02,
        help="Future return threshold for BUY label.",
    )

    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.02,
        help="Future return threshold for SELL label.",
    )

    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.0,
        help="Future return threshold for binary direction label.",
    )

    parser.add_argument(
        "--keep-all-forward-returns",
        action="store_true",
        help="Keep all future return columns. Not recommended for model training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    signal_df = build_signal_dataset(
        input_path=input_path,
        output_path=output_path,
        horizon=args.horizon,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        binary_threshold=args.binary_threshold,
        drop_leaky_columns=not args.keep_all_forward_returns,
    )

    summary = build_label_summary(signal_df)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()