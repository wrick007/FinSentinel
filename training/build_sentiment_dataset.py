import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import clean_text, logger, safe_str


DEFAULT_INPUT_DIR = RAW_DATA_DIR / "external" / "sentiment"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "sentiment_dataset.csv"


TEXT_COLUMN_CANDIDATES = [
    "text",
    "sentence",
    "headline",
    "title",
    "content",
    "summary",
    "news",
    "body",
    "tweet",
    "message",
    "comment",
    "description",
    "phrase",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "sentiment",
    "sentiment_label",
    "target",
    "class",
    "category",
    "polarity",
    "sentiment_score",
    "score",
    "Sentiment",
]

VALID_LABELS = {"positive", "negative", "neutral"}


def ensure_dirs() -> None:
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fix_encoding_text(text: Any) -> str:
    text = safe_str(text)

    if not text:
        return ""

    replacements = {
        "â€œ": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â": "",
        "�": "",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return clean_text(text)


def normalize_label(value: Any) -> str:
    raw = safe_str(value).strip().lower()

    if not raw:
        return ""

    mapping = {
        "positive": "positive",
        "pos": "positive",
        "bullish": "positive",
        "buy": "positive",
        "up": "positive",
        "good": "positive",
        "1": "positive",
        "+1": "positive",
        "2": "positive",
        "4": "positive",
        "5": "positive",

        "negative": "negative",
        "neg": "negative",
        "bearish": "negative",
        "sell": "negative",
        "down": "negative",
        "bad": "negative",
        "-1": "negative",

        "neutral": "neutral",
        "neu": "neutral",
        "hold": "neutral",
        "mixed": "neutral",
        "uncertain": "neutral",
        "0": "neutral",
        "3": "neutral",
    }

    if raw in mapping:
        return mapping[raw]

    try:
        number = float(raw)

        if number > 0:
            return "positive"

        if number < 0:
            return "negative"

        return "neutral"

    except Exception:
        return ""


def find_best_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {str(col).lower().strip(): col for col in columns}

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()

        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    for col in columns:
        lower = str(col).lower().strip()

        for candidate in candidates:
            if candidate.lower().strip() in lower:
                return col

    return None


def detect_text_label_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    columns = list(df.columns)

    text_col = find_best_column(columns, TEXT_COLUMN_CANDIDATES)
    label_col = find_best_column(columns, LABEL_COLUMN_CANDIDATES)

    if text_col is None and len(columns) >= 1:
        text_col = columns[0]

    if label_col is None and len(columns) >= 2:
        possible_label_cols = []

        for col in columns:
            try:
                unique_values = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.lower()
                    .head(5000)
                    .unique()
                    .tolist()
                )

                if len(unique_values) <= 20:
                    normalized = [normalize_label(value) for value in unique_values]
                    valid_count = sum(value in VALID_LABELS for value in normalized)

                    if valid_count > 0:
                        possible_label_cols.append(col)

            except Exception:
                continue

        if possible_label_cols:
            label_col = possible_label_cols[0]

    return text_col, label_col


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except Exception:
            continue

    try:
        return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip")
    except Exception as error:
        logger.warning("Could not read CSV %s: %s", path, error)
        return pd.DataFrame()


def parse_financial_phrasebank_txt(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    lines = []

    for encoding in encodings:
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except Exception:
            continue

    rows = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        text = ""
        label = ""

        if "@" in line:
            text, label = line.rsplit("@", 1)

        elif "\t" in line:
            text, label = line.rsplit("\t", 1)

        elif ";" in line:
            text, label = line.rsplit(";", 1)

        elif "," in line:
            try:
                parsed = next(csv.reader([line]))
                if len(parsed) >= 2:
                    text = ",".join(parsed[:-1])
                    label = parsed[-1]
            except Exception:
                text = ""
                label = ""

        text = fix_encoding_text(text)
        label = normalize_label(label)

        if text and label in VALID_LABELS:
            rows.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    if not rows:
        logger.warning(
            "No labelled rows detected in TXT file %s. Expected format like sentence@positive.",
            path.name,
        )

    return pd.DataFrame(rows)


def read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return read_csv_safely(path)

        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)

        if suffix == ".json":
            return pd.read_json(path)

        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)

        if suffix == ".txt":
            return parse_financial_phrasebank_txt(path)

        logger.warning("Unsupported file type skipped: %s", path)
        return pd.DataFrame()

    except Exception as error:
        logger.warning("Could not read file %s: %s", path, error)
        return pd.DataFrame()


def clean_single_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["text", "label", "source"])

    if {"text", "label"}.issubset(df.columns):
        result = pd.DataFrame()
        result["text"] = df["text"].apply(fix_encoding_text)
        result["label"] = df["label"].apply(normalize_label)
        result["source"] = source_name

    else:
        text_col, label_col = detect_text_label_columns(df)

        if text_col is None or label_col is None:
            logger.warning(
                "Skipping %s because text/label columns could not be detected. Columns: %s",
                source_name,
                list(df.columns),
            )
            return pd.DataFrame(columns=["text", "label", "source"])

        result = pd.DataFrame()
        result["text"] = df[text_col].apply(fix_encoding_text)
        result["label"] = df[label_col].apply(normalize_label)
        result["source"] = source_name

    result = result[result["text"].str.len() >= 8].copy()
    result = result[result["label"].isin(VALID_LABELS)].copy()

    result = result.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return result


def discover_files(input_dir: Path) -> List[Path]:
    supported = {".csv", ".parquet", ".pq", ".json", ".jsonl", ".txt"}

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return []

    files = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            files.append(path)

    return sorted(files)


def balance_dataset(
    df: pd.DataFrame,
    max_per_class: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if df.empty:
        return df

    frames = []

    for label, group in df.groupby("label"):
        if max_per_class is not None and len(group) > max_per_class:
            group = group.sample(max_per_class, random_state=random_state)

        frames.append(group)

    result = pd.concat(frames, ignore_index=True)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return result


def build_sentiment_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_per_class: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    ensure_dirs()

    files = discover_files(input_dir)

    if not files:
        raise FileNotFoundError(
            f"No sentiment files found in {input_dir}. "
            "Put CSV, Parquet, JSON, JSONL, or TXT files there first."
        )

    frames = []

    for file_path in files:
        logger.info("Reading sentiment file: %s", file_path)

        raw_df = read_file(file_path)

        if raw_df.empty:
            logger.warning("Skipping empty/unreadable file: %s", file_path.name)
            continue

        logger.info("Columns in %s: %s", file_path.name, list(raw_df.columns))

        cleaned_df = clean_single_dataset(
            raw_df,
            source_name=file_path.name,
        )

        if cleaned_df.empty:
            logger.warning("No usable labelled rows from %s", file_path.name)
            continue

        frames.append(cleaned_df)

        logger.info("Added %s rows from %s", len(cleaned_df), file_path.name)

    if not frames:
        raise ValueError("No usable labelled sentiment rows found after cleaning all files.")

    dataset = pd.concat(frames, ignore_index=True)

    dataset["text"] = dataset["text"].apply(fix_encoding_text)
    dataset["label"] = dataset["label"].apply(normalize_label)

    dataset = dataset[dataset["label"].isin(VALID_LABELS)].copy()
    dataset = dataset[dataset["text"].str.len() >= 8].copy()

    dataset = dataset.drop_duplicates(subset=["text"]).reset_index(drop=True)

    dataset = balance_dataset(
        dataset,
        max_per_class=max_per_class,
        random_state=random_state,
    )

    dataset = dataset[["text", "label", "source"]].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Saved sentiment dataset to %s", output_path)
    logger.info("Final shape: %s", dataset.shape)

    return dataset


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "label_distribution": {},
            "source_distribution": {},
        }

    return {
        "rows": int(len(df)),
        "label_distribution": df["label"].value_counts().to_dict(),
        "label_distribution_pct": (
            df["label"].value_counts(normalize=True) * 100
        ).round(2).to_dict(),
        "source_distribution": df["source"].value_counts().head(30).to_dict(),
    }


def print_summary(summary: Dict[str, Any], output_path: Path) -> None:
    print("\nFinSentinel Sentiment Dataset Summary")
    print("-------------------------------------")
    print(f"Rows: {summary['rows']}")

    print("\nLabel distribution:")
    for label, count in summary.get("label_distribution", {}).items():
        pct = summary.get("label_distribution_pct", {}).get(label, 0.0)
        print(f"  {label}: {count} rows ({pct:.2f}%)")

    print("\nTop sources:")
    for source, count in summary.get("source_distribution", {}).items():
        print(f"  {source}: {count}")

    print(f"\nSaved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean FinSentinel sentiment_dataset.csv from CSV, Parquet, JSON, JSONL, and Financial PhraseBank TXT files."
    )

    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing raw sentiment datasets.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class to balance very large datasets.",
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

    dataset = build_sentiment_dataset(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        max_per_class=args.max_per_class,
        random_state=args.random_state,
    )

    summary = build_summary(dataset)
    print_summary(summary, Path(args.output))


if __name__ == "__main__":
    main()