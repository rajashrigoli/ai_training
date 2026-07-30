import os
from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset, load_from_disk
import soundfile as sf
import librosa
from transformers import pipeline
import numpy as np
import gradio as gr
import soundfile as sf
import io

dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)

example = next(iter(dataset))
dataset_head = dataset.take(5)
list(dataset_head)
list(dataset_head)[0]
example

# Save audio to wav file for playback
output_path = "audio_sample.wav"
sf.write(output_path, example["audio"]["array"], example["audio"]["sampling_rate"])
dash_line = "-" * 100
print("\n\n")
print(dash_line)
print("Data preparation")
print(dash_line)
print(f"Audio saved to: {output_path}")

asr = pipeline(task="automatic-speech-recognition",model="distil-whisper/distil-small.en")
asr.feature_extractor.sampling_rate
example["audio"]["sampling_rate"]

# Convert audio to the correct format to avoid torchcodec issues
audio_data = example["audio"]["array"].astype(np.float32)
# Normalize if needed (values should be in [-1, 1] range)
if np.max(np.abs(audio_data)) > 1:
    audio_data = audio_data / np.max(np.abs(audio_data))

# Pass as dict with array and sampling_rate
audio_input = {
    "array": audio_data,
    "sampling_rate": example["audio"]["sampling_rate"]
}
result = asr(audio_input)
print("Transcription:", result["text"])
print(example["text"])

print("\n\n")
print(dash_line)
print("Build a shareable app with Gradio")
print(dash_line)
demo = gr.Blocks()

def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(filepath)
    return output["text"]


mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never")


file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never",
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )

demo.launch(share=True, server_port=int(os.environ.get('PORT1', 7860)))

demo.close()

print("\n\n")
print(dash_line)
print("Testing with longer audio files")
print(dash_line)

audio, sampling_rate = sf.read('narration_example.wav')
asr.feature_extractor.sampling_rate
asr(audio)
audio.shape

audio_transposed = np.transpose(audio)
audio_transposed.shape
audio_mono = librosa.to_mono(audio_transposed)

output_path = "audio_sample_mono.wav"
sf.write(output_path, audio_mono, sampling_rate)

asr(audio_mono)
asr.feature_extractor.sampling_rate
audio_16KHz = librosa.resample(audio_mono,
                               orig_sr=sampling_rate,
                               target_sr=16000)
asr(
    audio_16KHz,
    chunk_length_s=30, # 30 seconds
    batch_size=4,
    return_timestamps=True,
)["chunks"]
demo = gr.Blocks()
def transcribe_long_form(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(
      filepath,
      max_new_tokens=256,
      chunk_length_s=30,
      batch_size=8,
    )
    return output["text"]
mic_transcribe = gr.Interface(
    fn=transcribe_long_form,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never")

file_transcribe = gr.Interface(
    fn=transcribe_long_form,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never",
)
with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )
demo.launch(share=True, 
            server_port=int(os.environ.get('PORT1', 7861)))
demo.close()