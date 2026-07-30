from transformers.utils import logging
logging.set_verbosity_error()

import soundfile as sf
from datasets import load_dataset, load_from_disk
from datasets import Audio

from transformers import pipeline

dataset = load_dataset("ashraq/esc50")

audio_sample = dataset["train"][0]

print("Category:", audio_sample["category"])
print("Sampling rate:", audio_sample["audio"]["sampling_rate"])
print("Audio array shape:", audio_sample["audio"]["array"].shape)

# Save audio to wav file for playback
output_path = "audio_sample.wav"
sf.write(output_path, audio_sample["audio"]["array"], audio_sample["audio"]["sampling_rate"])
print(f"Audio saved to: {output_path}")

zero_shot_classifier = pipeline(task="zero-shot-audio-classification", model="laion/clap-htsat-unfused")

print("Audio sampling rate:", audio_sample["audio"]["sampling_rate"])
print("Zero-shot sampling rate:", zero_shot_classifier.feature_extractor.sampling_rate)

dataset = dataset.cast_column("audio", Audio(sampling_rate=48_000))
audio_sample = dataset["train"][0]

candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]
result = zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels)
print("Predicted label:", result[0]["label"])
candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]
result = zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels)
print("Predicted label:", result[0]["label"])

