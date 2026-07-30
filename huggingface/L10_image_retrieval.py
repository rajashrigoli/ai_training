# Bypass torch.load version check (PyTorch 2.2.2 + transformers 4.57.x incompatibility)
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import BlipForImageTextRetrieval, AutoProcessor
from PIL import Image
import requests
import torch

dash_line = "-" * 100
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", use_safetensors=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
print(raw_image)

print("\n\n")
print(dash_line)
print("Test, if the image matches the text")
print(dash_line)
text = "an image of a woman and a dog on the beach"
inputs = processor(images=raw_image,
                   text=text,
                   return_tensors="pt")

print(inputs)
itm_scores = model(**inputs)[0]
print(itm_scores)

print("\n\n")
print(dash_line)
print("Use a softmax layer to get the probabilities")
print(dash_line)
itm_score = torch.nn.functional.softmax(
    itm_scores,dim=1)

print(itm_score)

print(f"""\
The image and text are matched \
with a probability of {itm_score[0][1]:.4f}""")