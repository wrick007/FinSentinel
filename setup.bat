@echo off
setlocal

echo ==========================================
echo        FinSentinel Project Setup
echo ==========================================

cd /d "%~dp0"

echo.
echo Creating folder structure...

mkdir src 2>nul
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\news_cache 2>nul
mkdir models 2>nul
mkdir models\finbert 2>nul
mkdir models\signal_model 2>nul
mkdir outputs 2>nul
mkdir outputs\charts 2>nul
mkdir outputs\reports 2>nul
mkdir notebooks 2>nul
mkdir tests 2>nul

echo.
echo Removing .gitkeep files if any exist...

del /s /q .gitkeep 2>nul

echo.
echo Creating empty Python files...

type nul > src\__init__.py
type nul > src\config.py
type nul > src\sentiment.py
type nul > src\market_data.py
type nul > src\indicators.py
type nul > src\signal_engine.py
type nul > src\scraper.py
type nul > src\backtester.py
type nul > src\utils.py

type nul > tests\test_sentiment.py
type nul > tests\test_indicators.py
type nul > tests\test_signal_engine.py

echo.
echo Writing requirements.txt...

type nul > requirements.txt
>> requirements.txt echo gradio
>> requirements.txt echo transformers
>> requirements.txt echo torch
>> requirements.txt echo pandas
>> requirements.txt echo numpy
>> requirements.txt echo yfinance
>> requirements.txt echo feedparser
>> requirements.txt echo beautifulsoup4
>> requirements.txt echo requests
>> requirements.txt echo scikit-learn
>> requirements.txt echo matplotlib
>> requirements.txt echo plotly
>> requirements.txt echo ta
>> requirements.txt echo joblib
>> requirements.txt echo python-dotenv

echo.
echo Writing .gitignore...

type nul > .gitignore
>> .gitignore echo venv/
>> .gitignore echo .venv/
>> .gitignore echo env/
>> .gitignore echo .env
>> .gitignore echo.
>> .gitignore echo __pycache__/
>> .gitignore echo *.pyc
>> .gitignore echo *.pyo
>> .gitignore echo *.pyd
>> .gitignore echo .Python
>> .gitignore echo.
>> .gitignore echo .ipynb_checkpoints/
>> .gitignore echo.
>> .gitignore echo data/raw/
>> .gitignore echo data/news_cache/
>> .gitignore echo models/
>> .gitignore echo outputs/
>> .gitignore echo.
>> .gitignore echo .DS_Store
>> .gitignore echo Thumbs.db
>> .gitignore echo .vscode/
>> .gitignore echo .idea/

echo.
echo Writing README.md...

type nul > README.md
>> README.md echo # FinSentinel
>> README.md echo.
>> README.md echo FinSentinel is an explainable financial signal analysis app.
>> README.md echo.
>> README.md echo It combines FinBERT-based financial news sentiment, stock price data, technical indicators, risk filters, and Buy/Hold/Sell signal generation through a Gradio interface.
>> README.md echo.
>> README.md echo ## Project Structure
>> README.md echo.
>> README.md echo FinSentinel/
>> README.md echo - app.py
>> README.md echo - requirements.txt
>> README.md echo - README.md
>> README.md echo - .gitignore
>> README.md echo - src/
>> README.md echo - data/
>> README.md echo - models/
>> README.md echo - outputs/
>> README.md echo - notebooks/
>> README.md echo - tests/
>> README.md echo.
>> README.md echo ## Run Locally
>> README.md echo.
>> README.md echo Activate the virtual environment:
>> README.md echo.
>> README.md echo venv\Scripts\activate
>> README.md echo.
>> README.md echo Run the app:
>> README.md echo.
>> README.md echo python app.py

echo.
echo Writing starter app.py...

type nul > app.py
>> app.py echo import gradio as gr
>> app.py echo.
>> app.py echo.
>> app.py echo def analyze_signal(ticker, headline):
>> app.py echo     signal = "HOLD"
>> app.py echo     confidence = "50%%"
>> app.py echo     reason = "Setup complete. Core FinSentinel files will be added step by step."
>> app.py echo     return signal, confidence, reason
>> app.py echo.
>> app.py echo.
>> app.py echo with gr.Blocks(title="FinSentinel") as demo:
>> app.py echo     gr.Markdown("# FinSentinel")
>> app.py echo     gr.Markdown("Financial sentiment and Buy/Hold/Sell signal analysis app.")
>> app.py echo.
>> app.py echo     ticker = gr.Textbox(label="Ticker", value="AAPL")
>> app.py echo     headline = gr.Textbox(label="News headline or text", lines=4)
>> app.py echo.
>> app.py echo     analyze_btn = gr.Button("Analyze")
>> app.py echo.
>> app.py echo     signal = gr.Textbox(label="Signal")
>> app.py echo     confidence = gr.Textbox(label="Confidence")
>> app.py echo     reason = gr.Textbox(label="Reason")
>> app.py echo.
>> app.py echo     analyze_btn.click(
>> app.py echo         fn=analyze_signal,
>> app.py echo         inputs=[ticker, headline],
>> app.py echo         outputs=[signal, confidence, reason],
>> app.py echo     )
>> app.py echo.
>> app.py echo.
>> app.py echo if __name__ == "__main__":
>> app.py echo     demo.launch()

echo.
echo Checking Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python and add it to PATH.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...

if not exist venv (
    python -m venv venv
) else (
    echo venv already exists. Skipping creation.
)

if not exist "venv\Scripts\activate.bat" (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...

call venv\Scripts\activate.bat

echo.
echo Upgrading pip...

python -m pip install --upgrade pip

echo.
echo Installing requirements...

pip install -r requirements.txt

echo.
echo Initializing Git repository...

git --version >nul 2>&1
if errorlevel 1 (
    echo Git was not found. Skipping git init.
) else (
    if not exist .git (
        git init
    )
    git add .
    git commit -m "Initial FinSentinel Gradio project setup" || echo Git commit skipped. This is okay if there are no changes or Git identity is not configured.
)

echo.
echo ==========================================
echo FinSentinel setup completed successfully.
echo ==========================================
echo.
echo To activate venv later:
echo venv\Scripts\activate
echo.
echo To run app:
echo python app.py
echo.
pause