import sys
import os

# Add parent directory to path to import aiSetup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiSetup import print_llm_response, get_llm_response, get_chat_completion


import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, GenerationConfig


print("Summarize Dialogue without Prompt Engineering")
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

dash_line = "-" * 100
equal_line = "=" * 100

print(equal_line)
print(f"Dataset '{huggingface_dataset_name}' loaded successfully.")
print(equal_line)
example_indices = [40, 200]

for i, index in enumerate(example_indices):
    print(dash_line)
    print(f"Example {i + 1}:")
    print(dash_line)
    print("Input Dialogue:")
    print(dataset['test'][index]['dialogue'])
    print("Output Summary:")
    print(dataset['test'][index]['summary'])
    print(dash_line)


model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

sentence = "What time is it, Tom?"
sentence_encoded = tokenizer(sentence, return_tensors="pt")
sentence_decoded = tokenizer.decode(sentence_encoded["input_ids"][0], skip_special_tokens=True)


print("Encoded Sentence: ")
print(sentence_encoded["input_ids"][0])
print("\nDecoded Sentence:", sentence_decoded)


for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    inputs = tokenizer(dialogue, return_tensors="pt")
    output = tokenizer.decode(
        model.generate(inputs["input_ids"], max_new_tokens=50)[0], 
        skip_special_tokens=True
    )
    print(dash_line)
    print(f"Example {i + 1}:")
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')

print(equal_line)
print("Summarize Dialogue with an Instruction Prompt")

print("Zero Shot Inference with an Instruction Prompt")
print(equal_line)
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt_template = f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
    """
    inputs = tokenizer(prompt_template, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    print(dash_line)
    print(f"Example {i + 1}:")
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt_template}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - Zero Shot:\n{output}\n')
print(equal_line)
print("Zero Shot Inference with the Prompt Template from FLAN-T5")
print(equal_line)
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    prompt_template = f"""
    Dialogue:
    {dialogue}
    What was going on?
    """
    inputs = tokenizer(prompt_template, return_tensors="pt")
    output = tokenizer.decode(
        model.generate(inputs["input_ids"], max_new_tokens=50)[0], 
        skip_special_tokens=True
    )
    print(dash_line)
    print(f"Example {i + 1}:")
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt_template}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - Zero Shot:\n{output}\n')

print(equal_line)
print("Summarize Dialogue with One Shot and Few Shot Inference")
print(equal_line)

def make_prompt(example_indices_full, example_index_to_summarize):
    for i, index in enumerate(example_indices):
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']
        
        prompt = f"""
        Dialogue:
        {dialogue}
        What was going on?
        {summary}
        """

        dialogue = dataset['test'][example_index_to_summarize]['dialogue']
        prompt += f"""
        Dialogue:
        {dialogue}
        What was going on?
        {summary}
        """

        return prompt

print("One Shot Inference")
example_indices_full = [40]
example_index_to_summarize = 200
one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
print("One Shot Prompt:")
print(one_shot_prompt)

summary = dataset['test'][example_index_to_summarize]['summary']
inputs = tokenizer(one_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], max_new_tokens=50)[0], 
    skip_special_tokens=True
)
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')


print("Few Shot Inference")
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200
few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)
print("Few Shot Prompt:")
print(few_shot_prompt)

summary = dataset['test'][example_index_to_summarize]['summary']
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], max_new_tokens=50)[0], 
    skip_special_tokens=True
)
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')

print(equal_line)
print("Generative Configuration Parameters for Inference")
print("Gen Config with token limit of 50")
print(equal_line)
generation_config = GenerationConfig(max_new_tokens=50)
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], generation_config=generation_config)[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'MODEL GENERATION - WITH GEN CONFIG:\n{output}')
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")

print(equal_line)
print("Gen Config with token limit of 10")
print(equal_line)
generation_config = GenerationConfig(max_new_tokens=10)
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], generation_config=generation_config)[0],
    skip_special_tokens=True
)
print(dash_line)
print(f'MODEL GENERATION - WITH GEN CONFIG:\n{output}')
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")


print(equal_line)
print("Model Generation with Temperature 0.1")
generation_config = GenerationConfig(max_new_tokens=50, temperature=0.1)
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], generation_config=generation_config)[0],
    skip_special_tokens=True
)
print(dash_line)
print(f'MODEL GENERATION - WITH TEMPERATURE 0.1:\n{output}')
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
print(dash_line)

print(equal_line)
print("Model Generation with Temperature 0.5")
generation_config = GenerationConfig(max_new_tokens=50, temperature=0.5)
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], generation_config=generation_config)[0],
    skip_special_tokens=True
)
print(dash_line)
print(f'MODEL GENERATION - WITH TEMPERATURE 0.5:\n{output}')
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
print(dash_line)

print(equal_line)
print("Model Generation with Temperature 1.0")
generation_config = GenerationConfig(max_new_tokens=50, temperature=1.0)
inputs = tokenizer(few_shot_prompt, return_tensors="pt")
output = tokenizer.decode(
    model.generate(inputs["input_ids"], generation_config=generation_config)[0],
    skip_special_tokens=True
)
print(dash_line)
print(f'MODEL GENERATION - WITH TEMPERATURE 1.0:\n{output}')
print(dash_line)
print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
print(dash_line)