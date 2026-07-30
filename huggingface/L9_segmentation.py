from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
from PIL import Image
from helper import show_pipe_masks_on_image, show_mask_on_image
from transformers import SamModel, SamProcessor
import torch
import numpy as np

dash_line = "-" * 100
print("\n\n")
print(dash_line)
print("Mask Generation with SAM")
print(dash_line)

sam_pipe = pipeline("mask-generation", "Zigeng/SlimSAM-uniform-77", device="cpu", trust_remote_code=True)

raw_image = Image.open('sam_meta_llamas.png')
raw_image = raw_image.resize((720, 375))

output = sam_pipe(raw_image, points_per_batch=32)
# processed_image = show_pipe_masks_on_image(raw_image, output)
# processed_image.save("sam_meta_llamas_output.png")


print("\n\n")
print(dash_line)
print("Faster Inference: Infer an Image and a Single Point")
print(dash_line)

model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")
raw_image = Image.open('sam_meta_llamas.png')
raw_image.resize((720, 375))
input_points = [[[1600, 700]]]

inputs = processor(raw_image,
                 input_points=input_points,
                 return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)

print(len(predicted_masks))
predicted_mask = predicted_masks[0]
predicted_mask.shape
outputs.iou_scores


for i in range(3):
    show_mask_on_image(raw_image, predicted_mask[:, i])


print("\n\n")
print(dash_line)
print("Depth Estimation with DPT")
print(dash_line)
depth_estimator = pipeline(task="depth-estimation",
                          model="Intel/dpt-large",
                          trust_remote_code=True)

raw_image = Image.open('gradio_tamagochi_vienna.png')
raw_image.resize((806, 621))
output = depth_estimator(raw_image)

output["predicted_depth"].shape
output["predicted_depth"].unsqueeze(1).shape

prediction = torch.nn.functional.interpolate(
    output["predicted_depth"].unsqueeze(1),
    size=raw_image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

print("\n\n")
print("Depth Estimation: Prediction Shape and Image Size")
print(prediction)
print(dash_line)
print("Prediction Shape:", prediction.shape)
print("Image Size:", raw_image.size[::-1])

output = prediction.squeeze().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

print("\n\n")
print("Depth Estimation: Save Depth Image")
print(dash_line)
depth.save("gradio_tamagochi_vienna_depth.png")