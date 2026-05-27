from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NEWS_CACHE_DIR = DATA_DIR / "news_cache"

MODELS_DIR = BASE_DIR / "models"
FINBERT_DIR = MODELS_DIR / "finbert"
SIGNAL_MODEL_DIR = MODELS_DIR / "signal_model"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"


APP_NAME = "FinSentinel"
APP_DESCRIPTION = (
    "Explainable financial sentiment and Buy/Hold/Sell signal analysis app."
)

DEFAULT_TICKER = "AAPL"
DEFAULT_COMPANY_NAME = "Apple"
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

SUPPORTED_PERIODS = [
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
]

SUPPORTED_INTERVALS = [
    "1d",
    "1wk",
    "1mo",
]


FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
FINBERT_LOCAL_PATH = FINBERT_DIR / "saved_model"

USE_LOCAL_FINBERT = os.getenv("USE_LOCAL_FINBERT", "false").lower() == "true"

MAX_TEXT_LENGTH = 512
SENTIMENT_BATCH_SIZE = 8

SENTIMENT_LABEL_MAP = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}

SENTIMENT_CONFIDENCE_FLOOR = 0.34


BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.35"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "-0.35"))

STRONG_BUY_THRESHOLD = 0.60
STRONG_SELL_THRESHOLD = -0.60

MIN_NEWS_COUNT = 1
MAX_NEWS_ITEMS = 25

SENTIMENT_WEIGHT = 0.35
TREND_WEIGHT = 0.25
MOMENTUM_WEIGHT = 0.15
VOLUME_WEIGHT = 0.15
RISK_WEIGHT = 0.10

WEIGHT_SUM = (
    SENTIMENT_WEIGHT
    + TREND_WEIGHT
    + MOMENTUM_WEIGHT
    + VOLUME_WEIGHT
    + RISK_WEIGHT
)


SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
SMA_MAJOR_WINDOW = 200

EMA_FAST_WINDOW = 12
EMA_SLOW_WINDOW = 26
MACD_SIGNAL_WINDOW = 9

RSI_WINDOW = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

ATR_WINDOW = 14
VOLATILITY_WINDOW = 20
VOLUME_WINDOW = 20

BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2


LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.70

MAX_ACCEPTABLE_VOLATILITY = 0.08
MIN_AVG_VOLUME = 100000

EXTREME_GAP_THRESHOLD = 0.05
HIGH_SPREAD_THRESHOLD = 0.005


TRANSACTION_COST = 0.001
INITIAL_CAPITAL = 100000.0

BACKTEST_BUY_LEVEL = 0.35
BACKTEST_SELL_LEVEL = -0.35

RISK_FREE_RATE = 0.06


GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
NEWS_LOOKBACK_DAYS = 7

SOURCE_WEIGHTS = {
    "exchange_filing": 1.50,
    "reuters": 1.30,
    "bloomberg": 1.30,
    "financial_times": 1.25,
    "moneycontrol": 1.15,
    "economic_times": 1.10,
    "yahoo_finance": 1.00,
    "google_news": 0.95,
    "generic": 0.80,
}

DEFAULT_SOURCE_WEIGHT = 0.90


SIGNAL_LABELS = {
    "strong_buy": "STRONG BUY",
    "buy": "BUY",
    "hold": "HOLD",
    "sell": "SELL",
    "strong_sell": "STRONG SELL",
}

RISK_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}


HF_SPACE_MODE = os.getenv("SPACE_ID") is not None
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))


def ensure_directories() -> None:
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NEWS_CACHE_DIR,
        MODELS_DIR,
        FINBERT_DIR,
        SIGNAL_MODEL_DIR,
        OUTPUTS_DIR,
        CHARTS_DIR,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()