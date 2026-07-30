# Bypass torch.load version check (PyTorch 2.2.2 + transformers 4.57.x incompatibility)
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

import os
import gradio as gr
from transformers import pipeline

pipe = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-base")

def launch(input):
    out = pipe(input)
    return out[0]['generated_text']

iface = gr.Interface(launch,
                     inputs=gr.Image(type='pil'),
                     outputs="text")

iface.launch(share=True, server_port=int(os.environ.get('PORT1', 7860)))

iface.close()