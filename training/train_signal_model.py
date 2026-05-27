import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import logger, safe_float


DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "signal_dataset.csv"
DEFAULT_MODEL_DIR = MODELS_DIR / "signal_model"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "signal_model.pkl"
DEFAULT_REPORT_PATH = DEFAULT_MODEL_DIR / "training_report.json"
DEFAULT_FEATURE_IMPORTANCE_PATH = DEFAULT_MODEL_DIR / "feature_importance.csv"


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


PREFERRED_NUMERIC_FEATURES = [
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
]


PREFERRED_CATEGORICAL_FEATURES = [
    "ticker",
    "dominant_label",
]


def load_signal_dataset(input_path: Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Signal dataset not found: {input_path}. "
            "Run training/build_dataset.py and training/prepare_labels.py first."
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Signal dataset is empty.")

    required_columns = {"ticker", "date", "label", "label_id"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Signal dataset missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["label"] = df["label"].astype(str).str.upper()
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")

    df = df.dropna(subset=["ticker", "date", "label", "label_id"]).copy()
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["label_id"] = df["label_id"].astype(int)

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df


def remove_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = set()

    for col in df.columns:
        lower = col.lower()

        if col in DROP_COLUMNS:
            drop_cols.add(col)

        if lower.startswith(LEAKY_PREFIXES):
            drop_cols.add(col)

    return df.drop(columns=list(drop_cols), errors="ignore")


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    candidate_df = remove_leaky_features(df)

    protected = {"date"}

    candidate_df = candidate_df.drop(columns=list(protected), errors="ignore")

    numeric_features = [
        col
        for col in PREFERRED_NUMERIC_FEATURES
        if col in candidate_df.columns
    ]

    categorical_features = [
        col
        for col in PREFERRED_CATEGORICAL_FEATURES
        if col in candidate_df.columns
    ]

    extra_numeric = []

    for col in candidate_df.columns:
        if col in numeric_features or col in categorical_features:
            continue

        if pd.api.types.is_numeric_dtype(candidate_df[col]):
            extra_numeric.append(col)

    numeric_features = numeric_features + extra_numeric

    numeric_features = sorted(list(dict.fromkeys(numeric_features)))
    categorical_features = sorted(list(dict.fromkeys(categorical_features)))

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found.")

    return numeric_features, categorical_features


def make_time_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split empty dataframe.")

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    unique_dates = sorted(df["date"].dropna().unique())

    if len(unique_dates) < 5:
        split_idx = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df

    split_date_index = int(len(unique_dates) * (1.0 - test_size))
    split_date_index = max(1, min(split_date_index, len(unique_dates) - 1))

    split_date = unique_dates[split_date_index]

    train_df = df[df["date"] < split_date].copy()
    test_df = df[df["date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        split_idx = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.30,
    )


def get_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower().strip()

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="multiclass",
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        except Exception as error:
            logger.warning("LightGBM unavailable, using RandomForest instead: %s", error)
            model_name = "randomforest"

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception as error:
            logger.warning("XGBoost unavailable, using RandomForest instead: %s", error)
            model_name = "randomforest"

    if model_name == "logistic":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=-1,
            random_state=random_state,
        )

    if model_name == "gradientboosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=random_state,
        )

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def build_pipeline(
    model_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
    random_state: int = 42,
) -> Pipeline:
    scale_numeric = model_name.lower().strip() == "logistic"

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    model = get_model(
        model_name=model_name,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
    }

    if y_proba is not None:
        try:
            confidence = np.max(y_proba, axis=1)
            metrics["avg_prediction_confidence"] = round(float(np.mean(confidence)), 6)
        except Exception:
            metrics["avg_prediction_confidence"] = 0.0

    return metrics


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    labels = [0, 1, 2]
    target_names = [INVERSE_LABEL_MAP[label] for label in labels]

    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    labels = [0, 1, 2]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "labels": [INVERSE_LABEL_MAP[label] for label in labels],
        "matrix": matrix.tolist(),
    }


def get_transformed_feature_names(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        try:
            cat_pipeline = preprocessor.named_transformers_["cat"]
            onehot = cat_pipeline.named_steps["onehot"]
            cat_names = onehot.get_feature_names_out(categorical_features).tolist()
            feature_names.extend(cat_names)
        except Exception:
            feature_names.extend(categorical_features)

    return feature_names


def extract_feature_importance(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(
        pipeline=pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.mean(np.abs(coef), axis=0)

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    length = min(len(feature_names), len(importances))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names[:length],
            "importance": importances[:length],
        }
    )

    importance_df["importance"] = pd.to_numeric(
        importance_df["importance"],
        errors="coerce",
    ).fillna(0.0)

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


def build_training_package(
    pipeline: Pipeline,
    numeric_features: List[str],
    categorical_features: List[str],
    label_map: Dict[str, int],
    inverse_label_map: Dict[int, str],
    metrics: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    return {
        "pipeline": pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_columns": numeric_features + categorical_features,
        "label_map": label_map,
        "inverse_label_map": inverse_label_map,
        "metrics": metrics,
        "model_name": model_name,
        "version": "1.0.0",
    }


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=str)


def train_signal_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    feature_importance_path: Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
    model_name: str = "lightgbm",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    logger.info("Loading signal dataset from %s", input_path)

    df = load_signal_dataset(input_path)

    logger.info("Loaded signal dataset shape: %s", df.shape)

    numeric_features, categorical_features = infer_feature_columns(df)

    logger.info("Numeric features: %s", len(numeric_features))
    logger.info("Categorical features: %s", len(categorical_features))

    train_df, test_df = make_time_split(df, test_size=test_size)

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty data.")

    logger.info("Train shape: %s", train_df.shape)
    logger.info("Test shape: %s", test_df.shape)

    feature_columns = numeric_features + categorical_features

    X_train = train_df[feature_columns].copy()
    y_train = train_df["label_id"].astype(int).values

    X_test = test_df[feature_columns].copy()
    y_test = test_df["label_id"].astype(int).values

    pipeline = build_pipeline(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    logger.info("Training model: %s", model_name)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    y_train_proba = None
    y_test_proba = None

    if hasattr(pipeline, "predict_proba"):
        try:
            y_train_proba = pipeline.predict_proba(X_train)
            y_test_proba = pipeline.predict_proba(X_test)
        except Exception:
            y_train_proba = None
            y_test_proba = None

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    report = {
        "model_name": model_name,
        "input_path": str(input_path),
        "model_path": str(model_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(df)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_count": int(len(feature_columns)),
        "train_start_date": str(train_df["date"].min()),
        "train_end_date": str(train_df["date"].max()),
        "test_start_date": str(test_df["date"].min()),
        "test_end_date": str(test_df["date"].max()),
        "train_label_distribution": train_df["label"].value_counts().to_dict(),
        "test_label_distribution": test_df["label"].value_counts().to_dict(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classification_report": classification_report_dict(y_test, y_test_pred),
        "confusion_matrix": confusion_matrix_dict(y_test, y_test_pred),
    }

    training_package = build_training_package(
        pipeline=pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        label_map=LABEL_MAP,
        inverse_label_map=INVERSE_LABEL_MAP,
        metrics=report,
        model_name=model_name,
    )

    save_pickle(training_package, model_path)
    save_json(report, report_path)

    importance_df = extract_feature_importance(
        pipeline=pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    if not importance_df.empty:
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(feature_importance_path, index=False)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved report to %s", report_path)

    return {
        "model": pipeline,
        "package": training_package,
        "report": report,
        "feature_importance": importance_df,
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\nFinSentinel Signal Model Training Summary")
    print("-----------------------------------------")
    print(f"Model: {report.get('model_name')}")
    print(f"Total rows: {report.get('total_rows')}")
    print(f"Train rows: {report.get('train_rows')}")
    print(f"Test rows: {report.get('test_rows')}")
    print(f"Feature count: {report.get('feature_count')}")
    print(f"Train period: {report.get('train_start_date')} to {report.get('train_end_date')}")
    print(f"Test period: {report.get('test_start_date')} to {report.get('test_end_date')}")

    print("\nTrain label distribution:")
    for label, count in report.get("train_label_distribution", {}).items():
        print(f"  {label}: {count}")

    print("\nTest label distribution:")
    for label, count in report.get("test_label_distribution", {}).items():
        print(f"  {label}: {count}")

    print("\nTest metrics:")
    for key, value in report.get("test_metrics", {}).items():
        print(f"  {key}: {value}")

    print("\nConfusion matrix:")
    cm = report.get("confusion_matrix", {})
    labels = cm.get("labels", [])
    matrix = cm.get("matrix", [])

    print(f"Labels: {labels}")
    for row in matrix:
        print(row)

    print(f"\nSaved model: {report.get('model_path')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FinSentinel Buy/Hold/Sell signal model."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input signal dataset CSV path.",
    )

    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Output model pickle path.",
    )

    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON training report path.",
    )

    parser.add_argument(
        "--feature-importance-path",
        default=str(DEFAULT_FEATURE_IMPORTANCE_PATH),
        help="Output feature importance CSV path.",
    )

    parser.add_argument(
        "--model",
        default="lightgbm",
        choices=[
            "lightgbm",
            "xgboost",
            "randomforest",
            "gradientboosting",
            "logistic",
        ],
        help="Model type.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Time-based test split size.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = train_signal_model(
        input_path=Path(args.input),
        model_path=Path(args.model_path),
        report_path=Path(args.report_path),
        feature_importance_path=Path(args.feature_importance_path),
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print_report(result["report"])

    importance_df = result.get("feature_importance")

    if importance_df is not None and not importance_df.empty:
        print("\nTop 15 features:")
        print(importance_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()