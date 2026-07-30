import os
import sys

#import helper functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from helper import load_image_from_url, render_results_in_image, ignore_warnings, summarize_predictions_natural_language

from transformers import pipeline
from transformers.utils import logging
logging.set_verbosity_error()
from PIL import Image
import gradio as gr


dash_line = "-" * 100
print("\n\n")
print(dash_line)
print("Object Detection: Use the Pipeline")
print(dash_line)    
od_pipe = pipeline(task="object-detection",
                       model="facebook/detr-resnet-50",
                       trust_remote_code=True)

raw_image = Image.open("od_image.png")
raw_image = raw_image.resize((569, 491))

pipeline_output = od_pipe(raw_image)
processed_image = render_results_in_image(
    raw_image, 
    pipeline_output)

processed_image.save("od_image_output.png")


print("\n\n")
print(dash_line)
print(" Using `Gradio` as a Simple Interface")
print(dash_line)

def get_pipeline_prediction(pil_image):
    
    pipeline_output = od_pipe(pil_image)
    
    processed_image = render_results_in_image(pil_image,
                                            pipeline_output)
    return processed_image


demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image", 
                  type="pil"),
  outputs=gr.Image(label="Output image with predicted instances",
                   type="pil")
)
demo.launch(share=True, server_port=int(os.environ.get('PORT1', 7860)))
demo.close()

print("\n\n")
print(dash_line)
print("Generate Audio Narration of an Image")
print(dash_line)
text = summarize_predictions_natural_language(pipeline_output)
tts_pipe = pipeline(task="text-to-speech", 
                    model="kakao-enterprise/vits-ljs",
                    trust_remote_code=True)

narrated_text = tts_pipe(text)

print(narrated_text)