import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import FINBERT_MODEL_NAME, FINBERT_DIR, PROCESSED_DATA_DIR
from src.utils import clean_text, logger


DEFAULT_INPUT_PATH = PROCESSED_DATA_DIR / "sentiment_dataset.csv"
DEFAULT_OUTPUT_DIR = FINBERT_DIR / "fine_tuned_finbert"
DEFAULT_REPORT_PATH = FINBERT_DIR / "finbert_training_report.json"


LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class FinancialSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, Any], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels.astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {}

        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[idx])

        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label: Any) -> str:
    label = str(label).strip().lower()

    mapping = {
        "pos": "positive",
        "positive": "positive",
        "bullish": "positive",
        "buy": "positive",
        "up": "positive",
        "good": "positive",
        "1": "positive",
        "+1": "positive",
        "2": "positive",
        "4": "positive",
        "5": "positive",

        "neg": "negative",
        "negative": "negative",
        "bearish": "negative",
        "sell": "negative",
        "down": "negative",
        "bad": "negative",
        "-1": "negative",

        "neu": "neutral",
        "neutral": "neutral",
        "mixed": "neutral",
        "hold": "neutral",
        "0": "neutral",
        "3": "neutral",
    }

    return mapping.get(label, label)


def detect_text_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    possible_text_cols = [
        "text",
        "headline",
        "sentence",
        "content",
        "news",
        "title",
        "summary",
    ]

    possible_label_cols = [
        "label",
        "sentiment",
        "Sentiment",
        "target",
        "class",
        "category",
        "polarity",
    ]

    text_col = None
    label_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if text_col is None:
        text_col = df.columns[0]

    if label_col is None:
        if len(df.columns) < 2:
            raise ValueError("Could not detect label column.")
        label_col = df.columns[1]

    return text_col, label_col


def load_sentiment_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Sentiment dataset not found: {input_path}. "
            "Create data/processed/sentiment_dataset.csv with text,label columns."
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Sentiment dataset is empty.")

    text_col, label_col = detect_text_label_columns(df)

    result = pd.DataFrame()
    result["text"] = df[text_col].apply(clean_text)
    result["label"] = df[label_col].apply(normalize_label)

    result = result[result["text"].str.len() > 0].copy()
    result = result[result["label"].isin(LABEL_TO_ID.keys())].copy()
    result["label_id"] = result["label"].map(LABEL_TO_ID).astype(int)

    result = result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if result.empty:
        raise ValueError("No valid sentiment rows after cleaning.")

    class_counts = result["label"].value_counts().to_dict()

    for label in LABEL_TO_ID:
        if label not in class_counts:
            raise ValueError(f"Missing required label class: {label}")

    return result


def tokenize_dataset(tokenizer, texts, max_length: int = 128):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=False,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def make_training_arguments(
    output_dir: Path,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    use_fp16: bool,
) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_dir": str(output_dir / "logs"),
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "report_to": "none",
        "seed": seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"

    if "logging_strategy" in signature.parameters:
        kwargs["logging_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def calculate_class_weights(labels: np.ndarray) -> np.ndarray:
    classes = np.array([0, 1, 2])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels.astype(int),
    )

    return weights.astype(np.float32)


def build_report(
    trainer: Trainer,
    test_dataset: FinancialSentimentDataset,
    y_test: np.ndarray,
    output_dir: Path,
    dataset_summary: Dict[str, Any],
) -> Dict[str, Any]:
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    y_pred = np.argmax(logits, axis=1)

    return {
        "model_output_dir": str(output_dir),
        "dataset_summary": dataset_summary,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2],
        ).tolist(),
        "label_map": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def train_finbert(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    base_model: str = FINBERT_MODEL_NAME,
    test_size: float = 0.15,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 128,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.10,
    seed: int = 42,
    use_class_weights: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)

    logger.info("Loading sentiment dataset from %s", input_path)
    df = load_sentiment_dataset(input_path)

    logger.info("Sentiment dataset shape: %s", df.shape)
    logger.info("Label distribution: %s", df["label"].value_counts().to_dict())

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )

    logger.info("Train rows: %s", len(train_df))
    logger.info("Test rows: %s", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    train_encodings = tokenize_dataset(
        tokenizer=tokenizer,
        texts=train_df["text"].tolist(),
        max_length=max_length,
    )

    test_encodings = tokenize_dataset(
        tokenizer=tokenizer,
        texts=test_df["text"].tolist(),
        max_length=max_length,
    )

    train_dataset = FinancialSentimentDataset(
        train_encodings,
        train_df["label_id"].values,
    )

    test_dataset = FinancialSentimentDataset(
        test_encodings,
        test_df["label_id"].values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()

    training_args = make_training_arguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        seed=seed,
        use_fp16=use_fp16,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None

    if use_class_weights:
        class_weights = calculate_class_weights(train_df["label_id"].values)
        logger.info("Using class weights: %s", class_weights.tolist())

    trainer_kwargs = {
    "class_weights": class_weights,
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": test_dataset,
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
}

    trainer_signature = inspect.signature(Trainer.__init__)

    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)

    logger.info("Starting FinBERT fine-tuning...")
    trainer.train()

    logger.info("Saving fine-tuned FinBERT to %s", output_dir)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    dataset_summary = {
        "total_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "train_label_distribution": train_df["label"].value_counts().to_dict(),
        "test_label_distribution": test_df["label"].value_counts().to_dict(),
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "class_weights_used": use_class_weights,
        "cuda_available": torch.cuda.is_available(),
    }

    report = build_report(
        trainer=trainer,
        test_dataset=test_dataset,
        y_test=test_df["label_id"].values,
        output_dir=output_dir,
        dataset_summary=dataset_summary,
    )

    save_json(report, report_path)

    logger.info("Saved FinBERT report to %s", report_path)

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\nFinSentinel FinBERT Fine-tuning Summary")
    print("---------------------------------------")
    print(f"Output dir: {report.get('model_output_dir')}")
    print(f"Accuracy: {report.get('accuracy'):.4f}")
    print(f"Macro F1: {report.get('macro_f1'):.4f}")
    print(f"Weighted F1: {report.get('weighted_f1'):.4f}")

    print("\nConfusion matrix:")
    print("Labels: negative, neutral, positive")

    for row in report.get("confusion_matrix", []):
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune FinBERT on a labelled financial sentiment dataset."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path with text,label columns.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save fine-tuned FinBERT model.",
    )

    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to save JSON training report.",
    )

    parser.add_argument(
        "--base-model",
        default=FINBERT_MODEL_NAME,
        help="Base FinBERT model name.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Validation/test split size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = train_finbert(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report_path),
        base_model=args.base_model,
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
        use_class_weights=not args.no_class_weights,
    )

    print_report(report)


if __name__ == "__main__":
    main()