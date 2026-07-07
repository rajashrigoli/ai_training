import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import is_torch_available

# Prefer a local model folder if it exists, otherwise use the Hub model ID.
LOCAL_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "TinyLlama"
    / "TinyLlama-1.1B-Chat-v1.0"
)
HUB_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HUB_MODEL_ID

# Tokenizers work without PyTorch, so load this in all cases.
tokenizer = AutoTokenizer.from_pretrained(model_source)

model = AutoModelForCausalLM.from_pretrained(
    model_source,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

text_generator = pipeline(
        "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=20, 
    do_sample=False,
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

if text_generator is None:
    print("Skipping text generation because model/pipeline is unavailable.")
else:
    output = text_generator(prompt, max_new_tokens=50, do_sample=False,)
    print("Generated text:\n", output[0]["generated_text"])
