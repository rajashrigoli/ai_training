from transformers.utils import logging
logging.set_verbosity_error()

from transformers import BlipForQuestionAnswering, AutoProcessor
from PIL import Image
import requests


model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base", trust_remote_code=True)

image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(image_url, stream=True).raw)

question = "how many dogs are in the picture?"

inputs = processor(raw_image, question, return_tensors="pt")
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)
print(f"Answer: {answer}")
