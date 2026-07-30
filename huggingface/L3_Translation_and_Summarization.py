from transformers.utils import logging
logging.set_verbosity_error()

import torch
from transformers import pipeline

translator = pipeline(task="translation", model="facebook/nllb-200-distilled-600M", torch_dtype=torch.bfloat16)

text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""

translation = translator(text, src_lang="eng_Latn", tgt_lang="fra_Latn")[0]['translation_text']
print(translation)

translated_text = translator(text, src_lang="eng_Latn", tgt_lang="mar_Deva")[0]['translation_text']
print(translated_text)


print("\n\n\n")
print("Summarization Example")
print("---------------------")
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn", torch_dtype=torch.bfloat16)
text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
print(summary)