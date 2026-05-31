import os
import json
from typing import Any, Dict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.explanation import generate_rule_based_explanation


load_dotenv()


def build_gemma_prompt(signal_payload: Dict[str, Any]) -> str:
    payload_text = json.dumps(signal_payload, indent=2, default=str)

    return f"""
You are FinSentinel's reasoning layer.

Your task is to explain a model-generated BUY/HOLD/SELL stock signal.

Rules:
1. Use only the provided data.
2. Do not invent company news, prices, earnings, price targets, or external facts.
3. Do not give guaranteed financial advice.
4. Explain the signal in simple analyst-style language.
5. Mention both supporting reasons and risks.
6. Keep the answer under 180 words.

Data:
{payload_text}

Return exactly in this format:

Signal Summary:
Main Reasons:
Risk Factors:
Final View:
""".strip()


def generate_remote_gemma_reasoning(signal_payload: Dict[str, Any]) -> str:
    token = os.getenv("HF_TOKEN")
    model_name = os.getenv("GEMMA_MODEL", "google/gemma-2-2b-it")

    if not token:
        return (
            generate_rule_based_explanation(signal_payload)
            + "\n\nGemma Status:\nHF_TOKEN is missing, so fallback reasoning was used."
        )

    try:
        client = InferenceClient(
            model=model_name,
            token=token,
        )

        response = client.text_generation(
            prompt=build_gemma_prompt(signal_payload),
            max_new_tokens=250,
            temperature=0.25,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )

        if isinstance(response, str):
            return response.strip()

        return str(response)

    except Exception as error:
        fallback = generate_rule_based_explanation(signal_payload)

        return (
            f"{fallback}\n\n"
            f"Gemma Status:\n"
            f"Remote Gemma failed, so fallback reasoning was used.\n"
            f"Error: {error}"
        )