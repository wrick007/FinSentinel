import hashlib
import logging
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def setup_logger(name: str = "FinSentinel", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()


def ensure_dir(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_string() -> str:
    return now_utc().strftime("%Y%m%d_%H%M%S")


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    try:
        text = str(value)
    except Exception:
        return default

    if text.lower() in {"nan", "none", "null"}:
        return default

    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return default

        return result
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        result = int(float(value))
        return result
    except Exception:
        return default


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = safe_float(value, 0.0)
    return max(low, min(high, value))


def normalize_score(value: float, low: float = -1.0, high: float = 1.0) -> float:
    value = safe_float(value, 0.0)

    if high == low:
        return 0.0

    normalized = (value - low) / (high - low)
    normalized = (normalized * 2.0) - 1.0

    return clamp(normalized)


def sigmoid(value: float) -> float:
    value = clamp(value, -50, 50)
    return 1.0 / (1.0 + math.exp(-value))


def format_pct(value: Any, decimals: int = 2, default: str = "N/A") -> str:
    value = safe_float(value, float("nan"))

    if math.isnan(value):
        return default

    return f"{value * 100:.{decimals}f}%"


def format_number(value: Any, decimals: int = 2, default: str = "N/A") -> str:
    value = safe_float(value, float("nan"))

    if math.isnan(value):
        return default

    return f"{value:.{decimals}f}"


def format_large_number(value: Any, default: str = "N/A") -> str:
    value = safe_float(value, float("nan"))

    if math.isnan(value):
        return default

    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


def clean_text(text: Any) -> str:
    text = safe_str(text)

    if not text:
        return ""

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.strip()

    return text


def normalize_text(text: Any) -> str:
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9₹$%.,:;!?+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_text(text: Any, max_chars: int = 500, suffix: str = "...") -> str:
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - len(suffix)].rstrip() + suffix


def text_hash(text: Any) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_texts(texts: Sequence[Any]) -> List[str]:
    seen = set()
    unique_texts = []

    for item in texts:
        text = clean_text(item)
        key = text_hash(text)

        if text and key not in seen:
            unique_texts.append(text)
            seen.add(key)

    return unique_texts


def split_user_news_input(text: Any) -> List[str]:
    text = safe_str(text)

    if not text:
        return []

    lines = re.split(r"[\n\r]+", text)
    cleaned = []

    for line in lines:
        line = clean_text(line)

        if line:
            cleaned.append(line)

    if len(cleaned) <= 1 and "." in text:
        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned = [clean_text(part) for part in parts if clean_text(part)]

    return deduplicate_texts(cleaned)


def ticker_clean(ticker: Any) -> str:
    ticker = safe_str(ticker).upper()
    ticker = ticker.replace(" ", "")
    ticker = re.sub(r"[^A-Z0-9.\-_=]", "", ticker)
    return ticker


def company_clean(company_name: Any) -> str:
    return clean_text(company_name)


def is_valid_dataframe(df: Any, required_columns: Optional[Sequence[str]] = None) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            return False

    return True


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([safe_str(part) for part in col if safe_str(part)]).strip("_")
            for col in df.columns
        ]

    df.columns = [safe_str(col).strip() for col in df.columns]

    return df


def standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_yfinance_columns(df)

    rename_map = {}

    for col in df.columns:
        lower = col.lower()

        if lower.startswith("open"):
            rename_map[col] = "Open"
        elif lower.startswith("high"):
            rename_map[col] = "High"
        elif lower.startswith("low"):
            rename_map[col] = "Low"
        elif lower.startswith("close") and "adj" not in lower:
            rename_map[col] = "Close"
        elif "adj close" in lower or "adj_close" in lower:
            rename_map[col] = "Adj Close"
        elif lower.startswith("volume"):
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    return df


def safe_last_value(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
    dropna: bool = True,
) -> float:
    if df is None or df.empty or column not in df.columns:
        return default

    series = df[column]

    if dropna:
        series = series.dropna()

    if series.empty:
        return default

    return safe_float(series.iloc[-1], default)


def safe_last_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    return df.iloc[-1].to_dict()


def clean_numeric_series(series, fill_value: Optional[float] = None) -> pd.Series:
    if series is None:
        cleaned = pd.Series(dtype=float)
    elif isinstance(series, pd.DataFrame):
        if series.empty:
            cleaned = pd.Series(dtype=float)
        else:
            cleaned = pd.to_numeric(series.iloc[:, 0], errors="coerce")
    else:
        cleaned = pd.to_numeric(series, errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    if fill_value is not None:
        cleaned = cleaned.fillna(fill_value)

    return cleaned


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    result = pd.to_numeric(series, errors="coerce").pct_change(periods=periods)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")

    rolling_mean = series.rolling(window=window, min_periods=max(3, window // 3)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(3, window // 3)).std()

    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.replace([np.inf, -np.inf], np.nan)

    return zscore


def min_max_scale(value: float, min_value: float, max_value: float) -> float:
    value = safe_float(value, 0.0)
    min_value = safe_float(min_value, 0.0)
    max_value = safe_float(max_value, 1.0)

    if max_value == min_value:
        return 0.0

    scaled = (value - min_value) / (max_value - min_value)
    return max(0.0, min(1.0, scaled))


def confidence_from_score(score: float) -> float:
    score = abs(safe_float(score, 0.0))
    confidence = min(95.0, max(5.0, score * 100.0))
    return round(confidence, 2)


def risk_label_from_score(score: float) -> str:
    score = safe_float(score, 0.0)

    if score < 0.35:
        return "Low"

    if score < 0.70:
        return "Medium"

    return "High"


def safe_mean(values: Iterable[Any], default: float = 0.0) -> float:
    numbers = [safe_float(value, float("nan")) for value in values]
    numbers = [value for value in numbers if not math.isnan(value)]

    if not numbers:
        return default

    return float(np.mean(numbers))


def weighted_average(values: Sequence[Any], weights: Sequence[Any], default: float = 0.0) -> float:
    if not values or not weights or len(values) != len(weights):
        return default

    clean_values = np.array([safe_float(value, 0.0) for value in values], dtype=float)
    clean_weights = np.array([max(0.0, safe_float(weight, 0.0)) for weight in weights], dtype=float)

    total_weight = clean_weights.sum()

    if total_weight == 0:
        return default

    return float(np.average(clean_values, weights=clean_weights))


def time_decay_weight(hours_old: float, half_life_hours: float = 24.0) -> float:
    hours_old = max(0.0, safe_float(hours_old, 0.0))
    half_life_hours = max(1.0, safe_float(half_life_hours, 24.0))

    return float(0.5 ** (hours_old / half_life_hours))


def retry_call(
    func,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    logger_obj: Optional[logging.Logger] = None,
):
    logger_obj = logger_obj or logger

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return func()
        except exceptions as error:
            last_error = error
            logger_obj.warning(
                "Attempt %s/%s failed: %s",
                attempt,
                retries,
                error,
            )

            if attempt < retries:
                time.sleep(delay)
                delay *= backoff

    raise last_error


def dataframe_to_records(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    result = df.copy()

    if max_rows is not None:
        result = result.head(max_rows)

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.where(pd.notnull(result), None)

    return result.to_dict(orient="records")


def save_dataframe(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=index)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return path


def load_dataframe(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {suffix}")


def make_error_response(message: str, details: Optional[str] = None) -> Dict[str, Any]:
    return {
        "success": False,
        "message": clean_text(message),
        "details": clean_text(details),
        "timestamp": now_utc().isoformat(),
    }


def make_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "success": True,
        "message": clean_text(message),
        "data": data or {},
        "timestamp": now_utc().isoformat(),
    }