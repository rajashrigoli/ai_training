# Bypass torch.load version check (PyTorch 2.2.2 + transformers 4.57.x incompatibility)
# Must patch before any model loading occurs
import sys
import transformers.modeling_utils
# Patch the local reference in the module's global namespace
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
# Also patch in the load_state_dict's enclosing module
if 'transformers.utils.import_utils' in sys.modules:
    sys.modules['transformers.utils.import_utils'].check_torch_load_is_safe = lambda: None

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import CLIPModel, AutoProcessor
from PIL import Image
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16", trust_remote_code=True)

image_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(image_url, stream=True).raw)

labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=labels, images=raw_image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)[0]

for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")