import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import logger, safe_float


DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "signal_dataset.csv"
DEFAULT_MODEL_DIR = MODELS_DIR / "signal_model"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "evaluation"

LABEL_ORDER = [0, 1, 2]

INVERSE_LABEL_MAP = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

POSITION_MAP = {
    0: -1,
    1: 0,
    2: 1,
}


DROP_COLUMNS = {
    "label",
    "label_id",
    "binary_label",
    "target_return",
    "target_direction",
    "risk_adjusted_future_return",
    "target_horizon",
    "buy_threshold",
    "sell_threshold",
}

LEAKY_PREFIXES = (
    "future_close_",
    "future_return_",
)


def load_signal_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Signal dataset not found: {path}. Run prepare_labels.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Signal dataset is empty.")

    required = {"ticker", "date", "label", "label_id"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label"] = df["label"].astype(str).str.upper()
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")

    df = df.dropna(subset=["ticker", "date", "label", "label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def load_model_package(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as file:
        package = pickle.load(file)

    if isinstance(package, dict) and "pipeline" in package:
        return package

    return {
        "pipeline": package,
        "numeric_features": [],
        "categorical_features": [],
        "feature_columns": [],
        "model_name": model_path.stem,
        "inverse_label_map": INVERSE_LABEL_MAP,
    }


def get_model_files(model_path: Optional[Path], model_dir: Path) -> List[Path]:
    if model_path is not None:
        return [model_path]

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = sorted(model_dir.glob("*.pkl"))

    if not files:
        raise FileNotFoundError(f"No .pkl model files found in {model_dir}")

    return files


def remove_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = set()

    for col in df.columns:
        lower = col.lower()

        if col in DROP_COLUMNS:
            drop_cols.add(col)

        if lower.startswith(LEAKY_PREFIXES):
            drop_cols.add(col)

    return df.drop(columns=list(drop_cols), errors="ignore")


def resolve_feature_columns(
    package: Dict[str, Any],
    df: pd.DataFrame,
) -> List[str]:
    feature_columns = package.get("feature_columns") or []

    feature_columns = [col for col in feature_columns if col in df.columns]

    if feature_columns:
        return feature_columns

    candidate_df = remove_leaky_features(df)
    candidate_df = candidate_df.drop(columns=["date"], errors="ignore")

    feature_columns = [
        col
        for col in candidate_df.columns
        if col not in {"label", "label_id"}
    ]

    if not feature_columns:
        raise ValueError("No usable feature columns found for evaluation.")

    return feature_columns


def make_time_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> pd.DataFrame:
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = sorted(df["date"].dropna().unique())

    if len(unique_dates) < 5:
        split_idx = int(len(df) * (1.0 - test_size))
        return df.iloc[split_idx:].copy()

    split_date_index = int(len(unique_dates) * (1.0 - test_size))
    split_date_index = max(1, min(split_date_index, len(unique_dates) - 1))
    split_date = unique_dates[split_date_index]

    test_df = df[df["date"] >= split_date].copy()

    if test_df.empty:
        split_idx = int(len(df) * (1.0 - test_size))
        test_df = df.iloc[split_idx:].copy()

    return test_df


def predict_model(
    package: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    pipeline = package["pipeline"]
    feature_columns = resolve_feature_columns(package, df)

    X = df[feature_columns].copy()

    predictions = pipeline.predict(X)

    probabilities = None

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X)
        except Exception:
            probabilities = None

    return predictions, probabilities, feature_columns


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
    }

    if y_proba is not None:
        confidence = np.max(y_proba, axis=1)
        result["avg_prediction_confidence"] = round(float(np.mean(confidence)), 6)
        result["median_prediction_confidence"] = round(float(np.median(confidence)), 6)

    return result


def build_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    target_names = [INVERSE_LABEL_MAP[label] for label in LABEL_ORDER]

    return classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
    )

    return {
        "labels": [INVERSE_LABEL_MAP[label] for label in LABEL_ORDER],
        "matrix": matrix.tolist(),
    }


def prediction_distribution(y_pred: np.ndarray) -> Dict[str, int]:
    counts = pd.Series(y_pred).value_counts().to_dict()

    return {
        INVERSE_LABEL_MAP.get(int(label), str(label)): int(count)
        for label, count in counts.items()
    }


def actual_distribution(y_true: np.ndarray) -> Dict[str, int]:
    counts = pd.Series(y_true).value_counts().to_dict()

    return {
        INVERSE_LABEL_MAP.get(int(label), str(label)): int(count)
        for label, count in counts.items()
    }


def build_prediction_dataframe(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> pd.DataFrame:
    result = df.copy()

    result["pred_label_id"] = y_pred.astype(int)
    result["pred_label"] = result["pred_label_id"].map(INVERSE_LABEL_MAP)
    result["actual_label"] = result["label_id"].map(INVERSE_LABEL_MAP)

    if y_proba is not None:
        for idx, label_id in enumerate(LABEL_ORDER):
            if idx < y_proba.shape[1]:
                result[f"prob_{INVERSE_LABEL_MAP[label_id]}"] = y_proba[:, idx]

        result["pred_confidence"] = np.max(y_proba, axis=1)
    else:
        result["pred_confidence"] = np.nan

    return result


def calculate_strategy_metrics(
    pred_df: pd.DataFrame,
    allow_short: bool = False,
    transaction_cost: float = 0.001,
) -> Dict[str, Any]:
    if pred_df is None or pred_df.empty:
        return {}

    df = pred_df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "target_return" not in df.columns:
        return {
            "available": False,
            "message": "target_return column missing.",
        }

    df["target_return"] = pd.to_numeric(df["target_return"], errors="coerce")
    df["position"] = df["pred_label_id"].map(POSITION_MAP).fillna(0).astype(int)

    if not allow_short:
        df["position"] = df["position"].clip(lower=0)

    df["strategy_return"] = df["position"] * df["target_return"]

    df["trade"] = (
        df.groupby("ticker")["position"]
        .diff()
        .abs()
        .fillna(df["position"].abs())
    )

    df["strategy_return_after_cost"] = (
        df["strategy_return"] - df["trade"] * transaction_cost
    )

    valid_returns = df["strategy_return_after_cost"].dropna()

    if valid_returns.empty:
        return {
            "available": False,
            "message": "No valid strategy returns.",
        }

    total_return = float((1.0 + valid_returns).prod() - 1.0)
    mean_return = float(valid_returns.mean())
    std_return = float(valid_returns.std())

    sharpe_like = 0.0

    if std_return > 0:
        sharpe_like = float(mean_return / std_return)

    active_trades = df[df["position"] != 0].copy()

    win_rate = 0.0

    if not active_trades.empty:
        win_rate = float((active_trades["strategy_return_after_cost"] > 0).mean())

    buy_count = int((df["pred_label"] == "BUY").sum())
    sell_count = int((df["pred_label"] == "SELL").sum())
    hold_count = int((df["pred_label"] == "HOLD").sum())

    benchmark_return = float((1.0 + df["target_return"].dropna()).prod() - 1.0)

    return {
        "available": True,
        "total_strategy_return": round(total_return, 6),
        "benchmark_return": round(benchmark_return, 6),
        "excess_return": round(total_return - benchmark_return, 6),
        "avg_strategy_return": round(mean_return, 6),
        "strategy_return_std": round(std_return, 6),
        "sharpe_like_ratio": round(sharpe_like, 6),
        "active_trade_rows": int(len(active_trades)),
        "win_rate_on_active_rows": round(win_rate, 6),
        "buy_predictions": buy_count,
        "sell_predictions": sell_count,
        "hold_predictions": hold_count,
        "long_only": not allow_short,
        "transaction_cost": transaction_cost,
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def evaluate_single_model(
    model_path: Path,
    dataset: pd.DataFrame,
    output_dir: Path,
    test_size: float = 0.20,
    evaluate_full_dataset: bool = False,
    allow_short: bool = False,
    transaction_cost: float = 0.001,
) -> Dict[str, Any]:
    logger.info("Evaluating model: %s", model_path)

    package = load_model_package(model_path)

    if evaluate_full_dataset:
        eval_df = dataset.copy()
    else:
        eval_df = make_time_test_split(dataset, test_size=test_size)

    if eval_df.empty:
        raise ValueError("Evaluation dataframe is empty.")

    y_true = eval_df["label_id"].astype(int).values

    y_pred, y_proba, feature_columns = predict_model(package, eval_df)

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    report = build_classification_report(y_true, y_pred)
    matrix = build_confusion_matrix(y_true, y_pred)

    pred_df = build_prediction_dataframe(eval_df, y_pred, y_proba)

    strategy_metrics = calculate_strategy_metrics(
        pred_df=pred_df,
        allow_short=allow_short,
        transaction_cost=transaction_cost,
    )

    model_name = package.get("model_name") or model_path.stem

    evaluation = {
        "model_name": model_name,
        "model_path": str(model_path),
        "rows_evaluated": int(len(eval_df)),
        "start_date": str(eval_df["date"].min()),
        "end_date": str(eval_df["date"].max()),
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "actual_distribution": actual_distribution(y_true),
        "prediction_distribution": prediction_distribution(y_pred),
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "strategy_metrics": strategy_metrics,
    }

    safe_model_name = model_path.stem.replace(" ", "_")

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{safe_model_name}_evaluation_report.json"
    predictions_path = output_dir / f"{safe_model_name}_predictions.csv"
    confusion_path = output_dir / f"{safe_model_name}_confusion_matrix.csv"

    save_json(evaluation, report_path)
    pred_df.to_csv(predictions_path, index=False)

    confusion_df = pd.DataFrame(
        matrix["matrix"],
        index=matrix["labels"],
        columns=matrix["labels"],
    )
    confusion_df.to_csv(confusion_path)

    evaluation["report_path"] = str(report_path)
    evaluation["predictions_path"] = str(predictions_path)
    evaluation["confusion_matrix_path"] = str(confusion_path)

    return evaluation


def compare_models(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for evaluation in evaluations:
        metrics = evaluation.get("metrics", {})
        strategy = evaluation.get("strategy_metrics", {})

        rows.append(
            {
                "model_name": evaluation.get("model_name"),
                "model_path": evaluation.get("model_path"),
                "rows_evaluated": evaluation.get("rows_evaluated"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "avg_prediction_confidence": metrics.get("avg_prediction_confidence"),
                "total_strategy_return": strategy.get("total_strategy_return"),
                "benchmark_return": strategy.get("benchmark_return"),
                "excess_return": strategy.get("excess_return"),
                "win_rate_on_active_rows": strategy.get("win_rate_on_active_rows"),
                "active_trade_rows": strategy.get("active_trade_rows"),
            }
        )

    comparison_df = pd.DataFrame(rows)

    if not comparison_df.empty and "macro_f1" in comparison_df.columns:
        comparison_df = comparison_df.sort_values(
            by=["macro_f1", "balanced_accuracy", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)

    return comparison_df


def print_evaluation_summary(evaluation: Dict[str, Any]) -> None:
    print("\nModel Evaluation Summary")
    print("------------------------")
    print(f"Model: {evaluation.get('model_name')}")
    print(f"Rows evaluated: {evaluation.get('rows_evaluated')}")
    print(f"Period: {evaluation.get('start_date')} to {evaluation.get('end_date')}")
    print(f"Feature count: {evaluation.get('feature_count')}")

    print("\nActual distribution:")
    for label, count in evaluation.get("actual_distribution", {}).items():
        print(f"  {label}: {count}")

    print("\nPrediction distribution:")
    for label, count in evaluation.get("prediction_distribution", {}).items():
        print(f"  {label}: {count}")

    print("\nClassification metrics:")
    for key, value in evaluation.get("metrics", {}).items():
        print(f"  {key}: {value}")

    print("\nConfusion matrix:")
    cm = evaluation.get("confusion_matrix", {})
    print(f"Labels: {cm.get('labels', [])}")
    for row in cm.get("matrix", []):
        print(row)

    strategy = evaluation.get("strategy_metrics", {})

    if strategy.get("available"):
        print("\nStrategy-style check:")
        print(f"  Total strategy return: {strategy.get('total_strategy_return')}")
        print(f"  Benchmark return: {strategy.get('benchmark_return')}")
        print(f"  Excess return: {strategy.get('excess_return')}")
        print(f"  Win rate on active rows: {strategy.get('win_rate_on_active_rows')}")
        print(f"  Active trade rows: {strategy.get('active_trade_rows')}")

    print(f"\nSaved report: {evaluation.get('report_path')}")
    print(f"Saved predictions: {evaluation.get('predictions_path')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinSentinel trained signal models."
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to signal_dataset.csv.",
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to one model .pkl file. If omitted, all .pkl files in model-dir are evaluated.",
    )

    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model .pkl files.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for evaluation reports.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Time-based test split size.",
    )

    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluate on the full dataset instead of the time-based test split.",
    )

    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow SELL predictions to act as short positions in strategy check.",
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost used in strategy-style check.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    model_path = Path(args.model_path) if args.model_path else None

    dataset = load_signal_dataset(dataset_path)

    model_files = get_model_files(
        model_path=model_path,
        model_dir=model_dir,
    )

    evaluations = []

    for path in model_files:
        try:
            evaluation = evaluate_single_model(
                model_path=path,
                dataset=dataset,
                output_dir=output_dir,
                test_size=args.test_size,
                evaluate_full_dataset=args.full_dataset,
                allow_short=args.allow_short,
                transaction_cost=args.transaction_cost,
            )

            evaluations.append(evaluation)
            print_evaluation_summary(evaluation)

        except Exception as error:
            logger.exception("Failed to evaluate %s: %s", path, error)

    if not evaluations:
        raise RuntimeError("No models were evaluated successfully.")

    comparison_df = compare_models(evaluations)

    comparison_path = output_dir / "model_comparison.csv"
    comparison_json_path = output_dir / "model_comparison.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)

    save_json(
        {
            "models_evaluated": len(evaluations),
            "comparison": comparison_df.to_dict(orient="records"),
        },
        comparison_json_path,
    )

    print("\nModel Comparison")
    print("----------------")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()