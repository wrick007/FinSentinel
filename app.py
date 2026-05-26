import gradio as gr


def analyze_signal(ticker, headline):
    signal = "HOLD"
    confidence = "50%"
    reason = "Setup complete. Core FinSentinel files will be added step by step."
    return signal, confidence, reason


with gr.Blocks(title="FinSentinel") as demo:
    gr.Markdown("# FinSentinel")
    gr.Markdown("Financial sentiment and Buy/Hold/Sell signal analysis app.")

    ticker = gr.Textbox(label="Ticker", value="AAPL")
    headline = gr.Textbox(label="News headline or text", lines=4)

    analyze_btn = gr.Button("Analyze")

    signal = gr.Textbox(label="Signal")
    confidence = gr.Textbox(label="Confidence")
    reason = gr.Textbox(label="Reason")

    analyze_btn.click(
        fn=analyze_signal,
        inputs=[ticker, headline],
        outputs=[signal, confidence, reason],
    )


if __name__ == "__main__":
    demo.launch()
