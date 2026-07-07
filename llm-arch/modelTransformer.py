import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import is_torch_available


def print_runtime_diagnostics() -> None:
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {python_version}")
    print(f"Torch available: {is_torch_available()}")
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN is not set. Downloads may be slower and rate-limited.")

# Prefer a local model folder if it exists, otherwise use the Hub model ID.
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "microsoft"
    / "Phi-3-mini-4k-instruct"
)
HUB_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HUB_MODEL_ID

# Tokenizers work without PyTorch, so load this in all cases.
tokenizer = AutoTokenizer.from_pretrained(model_source)

model = None
text_generator = None

print_runtime_diagnostics()

if is_torch_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    text_generator = pipeline(
         "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=50, 
        do_sample=False,
    )
else:
    print("PyTorch is not installed. Loaded tokenizer only; model/pipeline are unavailable.")
    print("To enable text generation, create a Python 3.11/3.12 venv and install torch.")
    print("Example: python3.12 -m venv .venv312 && source .venv312/bin/activate && python -m pip install torch transformers")


prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

if text_generator is None:
    print("Skipping text generation because model/pipeline is unavailable.")
else:
    output = text_generator(prompt, use_cache=False)
    print("Generated text:\n", output[0]["generated_text"])
