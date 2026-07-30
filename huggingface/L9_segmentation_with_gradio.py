import os
import gradio as gr
from transformers import pipeline
import torch
import numpy as np
from PIL import Image

depth_estimator = pipeline("depth-estimation")

def launch(input_image):
    out = depth_estimator(input_image)

    # resize the prediction
    prediction = torch.nn.functional.interpolate(
        out["predicted_depth"].unsqueeze(0).unsqueeze(0),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # normalize the prediction
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

iface = gr.Interface(launch, 
                     inputs=gr.Image(type='pil'), 
                     outputs=gr.Image(type='pil'))


iface.launch(share=True, server_port=int(os.environ.get('PORT1', 7860)))
iface.close()