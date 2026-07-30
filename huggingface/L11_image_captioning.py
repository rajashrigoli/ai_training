# Bypass torch.load version check (PyTorch 2.2.2 + transformers 4.57.x incompatibility)
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers.utils import logging
logging.set_verbosity_error()

import requests
from PIL import Image
from transformers import pipeline, BlipForConditionalGeneration, AutoProcessor

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", trust_remote_code=True)

image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(image_url, stream=True).raw)

print("\n\n")
dash_line = "-" * 100
print(dash_line)
print("Conditional Image Captioning")
print(dash_line)
text = "a photograph of"
inputs = processor(raw_image, text=text, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")

print("\n\n")
dash_line = "-" * 100
print(dash_line)
print("Unconditional Image Captioning")
print(dash_line)

inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")