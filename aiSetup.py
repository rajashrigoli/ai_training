# import gradio as gr
import os
import json

from openai import OpenAI, NotFoundError, BadRequestError
from dotenv import load_dotenv

import random

#Get the OpenAI API key from the .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '', '.env'), override=True)

API_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")

# Set up the OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

AI_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GROQ_MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"
DEPRECATED_GROQ_MODERATION_MODELS = {
    "meta-llama/llama-guard-4-12b": DEFAULT_GROQ_MODERATION_MODEL,
    "llama-guard-3-8b": DEFAULT_GROQ_MODERATION_MODEL,
}
MODERATION_MODEL = os.getenv("MODERATION_MODEL", DEFAULT_GROQ_MODERATION_MODEL)

def _supports_moderation():
    """Return True when the configured provider supports /moderations."""
    return "api.groq.com" not in API_BASE_URL


def _moderate_with_groq_guard(text, model=MODERATION_MODEL):
    """Use a Groq-hosted guard model to classify text and return categories."""
    def _clamp_score(value, default):
        try:
            numeric = float(value)
            return max(0.0, min(1.0, numeric))
        except (TypeError, ValueError):
            return default

    def _extract_json_payload(raw_output):
        if not isinstance(raw_output, str):
            return None

        candidate = raw_output.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

        # Last resort: attempt to parse a JSON object embedded in free text.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    selected_model = DEPRECATED_GROQ_MODERATION_MODELS.get(model, model)
    lowered_text = text.lower()
    threat_markers = [
        "i know where you live",
        "i'm coming for you",
        "im coming for you",
        "watch your back",
        "i will kill you",
        "i'll kill you",
        "i am going to hurt you",
        "i'm going to hurt you",
        "i will find you",
        "coming for you",
    ]
    if any(marker in lowered_text for marker in threat_markers):
        return {
            "flagged": True,
            "label": "BLOCK",
            "score": 1.0,
            "categories": ["threat", "violence"],
            "reason": "Detected explicit threatening language.",
            "raw": "rule_based_threat_detection",
        }

    prompt = (
        "Classify the following text for safety risk. "
        "Return strict JSON with keys: decision, score, categories, reason. "
        "decision must be ALLOW or BLOCK. categories must be a JSON array of strings. "
        "score must be a float between 0 and 1 indicating confidence that content should be BLOCKED. "
        "Allowed category labels: threat, violence, harassment, hate, self_harm, sexual, illegal, other. "
        "If no unsafe category applies, return an empty array.\n\n"
        f"TEXT:\n{text}"
    )
    try:
        completion = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    except BadRequestError as err:
        # Fallback when a configured moderation model is deprecated.
        if "model_decommissioned" in str(err) and selected_model != DEFAULT_GROQ_MODERATION_MODEL:
            completion = client.chat.completions.create(
                model=DEFAULT_GROQ_MODERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        else:
            raise
    output = (completion.choices[0].message.content or "").strip()

    parsed = _extract_json_payload(output)

    if isinstance(parsed, dict):
        decision = str(parsed.get("decision", "ALLOW")).upper().strip()
        score = _clamp_score(parsed.get("score"), 1.0 if decision == "BLOCK" else 0.0)
        categories = parsed.get("categories", [])
        if not isinstance(categories, list):
            categories = []
        categories = [str(item) for item in categories]
        reason = str(parsed.get("reason", ""))
    else:
        # Fallback parser for non-JSON model outputs.
        normalized = output.upper()
        if normalized.replace(".", "", 1).isdigit():
            score = _clamp_score(normalized, 0.0)
            blocked = score >= 0.5
            return {
                "flagged": blocked,
                "label": "BLOCK" if blocked else "ALLOW",
                "score": score,
                "categories": ["other"] if blocked else [],
                "reason": f"Numeric safety score: {score}",
                "raw": output,
            }
        decision = "BLOCK" if (
            normalized.startswith("BLOCK")
            or normalized.startswith("UNSAFE")
            or "UNSAFE" in normalized
            or "THREAT" in normalized
            or "VIOLENCE" in normalized
            or "VIOLATION" in normalized
        ) else "ALLOW"
        score = 0.9 if decision == "BLOCK" else 0.0
        categories = ["other"] if decision == "BLOCK" else []
        reason = output

    blocked = decision == "BLOCK"
    return {
        "flagged": blocked,
        "label": decision,
        "score": score,
        "categories": categories,
        "reason": reason,
        "raw": output,
    }


def print_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then prints the response of the model.
    """
    llm_response = get_llm_response(prompt)
    print(llm_response)


def get_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then saves the response of the model as
    a string.
    """
    try:
        if not isinstance(prompt, str):
            raise ValueError("Input must be a string enclosed in quotes.")
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful but terse AI assistant who gets straight to the point.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        return response
    except TypeError as e:
        print("Error:", str(e))


def get_chat_completion(prompt, history=""):
    history_string = "\n\n".join(["\n".join(turn) for turn in history])
    prompt_with_history = f"{history_string}\n\n{prompt}"
    completion = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful but terse AI assistant who gets straight to the point.",
            },
            {"role": "user", "content": prompt_with_history},
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content
    return response

def get_completion_from_messages(messages, model=AI_MODEL, temperature=0.0, max_tokens=500):
    """This function takes as input a list of messages, which must be a list of dictionaries with
    'role' and 'content' keys, and passes it to OpenAI's GPT3.5 model. The function then saves the
    response of the model as a string.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        response = completion.choices[0].message.content
        return response
    except TypeError as e:
        print("Error:", str(e))
        return None

def get_completion_and_token_count(messages, model=AI_MODEL, temperature=0.0, max_tokens=500):
    """This function takes as input a list of messages, which must be a list of dictionaries with
    'role' and 'content' keys, and passes it to OpenAI's GPT3.5 model. The function then saves the
    response of the model as a string and returns the response along with the token count.
    """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        response = completion.choices[0].message.content
        token_count = completion.usage.total_tokens
        return response, token_count
    except TypeError as e:
        print("Error:", str(e))
        return None, 0


def get_completion_from_messages_with_moderation(
    messages,
    chat_model=AI_MODEL,
    moderation_model=MODERATION_MODEL,
    temperature=0.0,
    max_tokens=500,
):
    """This function takes as input a list of messages, which must be a list of dictionaries with
    'role' and 'content' keys, and passes it to OpenAI's GPT3.5 model. The function then saves the
    response of the model as a string and returns the response along with the token count.
    """
    try:
        # Groq uses guard models via chat completions for moderation.
        if "api.groq.com" in API_BASE_URL:
            for message in messages:
                if message['role'] == 'user':
                    moderation_response = _moderate_with_groq_guard(
                        text=message['content'],
                        model=moderation_model,
                    )
                    if moderation_response["flagged"]:
                        return "Input flagged by moderation.", 0
        elif _supports_moderation():
            try:
                for message in messages:
                    if message['role'] == 'user':
                        moderation_response = client.moderations.create(
                            model=MODERATION_MODEL,
                            input=message['content']
                        )
                        if moderation_response.results[0].flagged:
                            return "Input flagged by moderation.", 0
            except NotFoundError:
                # Some OpenAI-compatible providers do not support this endpoint.
                pass

        completion = client.chat.completions.create(
            model=chat_model,
            messages=messages,
            temperature=temperature,
        )
        response = completion.choices[0].message.content
        token_count = completion.usage.total_tokens
        return response, token_count
    except TypeError as e:
        print("Error:", str(e))
        return None, 0

def getModerationOutput(moderateText:str = None):
    sample_text = moderateText if moderateText is not None else f"""
            Here's the plan.  We get the warhead,
            and we hold the world ransom...
            ...FOR ONE MILLION DOLLARS!
            """

    if "api.groq.com" in API_BASE_URL:
        moderation_output = _moderate_with_groq_guard(sample_text, MODERATION_MODEL)
        print(moderation_output)
        return moderation_output

    if not _supports_moderation():
        print("Moderation endpoint is not supported by the configured provider.")
        return None

    try:
        response = client.moderations.create(
            model=MODERATION_MODEL,
            input=sample_text
        )
        moderation_output = response["results"][0]
        print(moderation_output)
        return moderation_output
    except NotFoundError:
        print("Moderation endpoint is unavailable for the current provider.")
        return None