from utils import (
    generate_with_single_input, 
    generate_with_multiple_input, 
    generate_params_dict,
    check_if_outfit_or_supplement,
    decide_if_technical_or_creative,
    answer_query,
    generate_system_call
)

from pydantic import BaseModel, validator, conint, Field
from typing import Literal, Union, Optional, List
import json

dash_line = "-" * 100
print(dash_line)
print("1 - Text Classification with LLMs")
print(dash_line)


# Testing a simple query
query = "Give me the available vitamins supplement you have in your catalogue."
result = generate_with_single_input(check_if_outfit_or_supplement(query), max_tokens = 2)
print(result)

# ASCII color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

queries = [
    {"query": "Where can I buy whey protein?", "label": "Nutritional"},
    {"query": "Recommended vitamins for winter", "label": "Nutritional"},
    {"query": "Latest fashion for women's dresses", "label": "Outfit"},
    {"query": "Comfortable sneakers for daily use", "label": "Outfit"},
    {"query": "Best energy bars for athletes", "label": "Nutritional"},
    {"query": "Trendy accessories for men", "label": "Outfit"},
    {"query": "Low-carb diet food options", "label": "Nutritional"},
    {"query": "What supplements help with muscle recovery?", "label": "Nutritional"},
    {"query": "Casual wear that supports healthy living", "label": "Outfit"}
]

for item in queries:
    query = item["query"]
    prompt = check_if_outfit_or_supplement(query)
    expected_label = item["label"]
    response = generate_with_single_input(prompt, max_tokens = 2)
    result = response['content']
    
    # Determine color based on comparison
    if result == expected_label:
        color = GREEN
    else:
        color = RED

    print(f"Query: {query}\nResult: {result}\nExpected: {color}{expected_label}{RESET}\n")


print("\n\n")
print(dash_line)
print("2 - Parameter Setting Based on Tasks")
print(dash_line)


queries = ["What is Pi-hole?", 
           "Suggest to me three places to visit in South America"]
for query in queries:
    label =decide_if_technical_or_creative(query)
    print(f"Query: {query}, label: {label}")


queries = ["What is Pi-hole?", 
           "Suggest to me three places to visit in South America"]
for query in queries:
    result = answer_query(query)
    print(f"Query: {query}\nAnswer: {result}\n\n#######\n")


print("\n\n")
print(dash_line)
print("3 - Guiding the LLM to Output Specific Object")
print(dash_line)    

print("\n\n")
print("3.1 The Old-Fashioned Way")
print(dash_line)

print(generate_system_call("Play a chill playlist very loud"))
print(generate_system_call("I'm tired today, please make my living room a very cozy ambient, it is really cold today too."))

print("\n\n")
print("3.2 Using LLM structured output parameter")
print(dash_line)

# Define the schema for the output
class VoiceNote(BaseModel):
    title: str = Field(description="A title for the voice note")
    summary: str = Field(description="A short one sentence summary of the voice note.")
    actionItems: list[str] = Field(
        description="A list of action items from the voice note"
    )

transcript = (
        "Good morning! It's 7:00 AM, and I'm just waking up. Today is going to be a busy day, "
        "so let's get started. First, I need to make a quick breakfast. I think I'll have some "
        "scrambled eggs and toast with a cup of coffee. While I'm cooking, I'll also check my "
        "emails to see if there's anything urgent."
    )


messages=[
            {
                "role": "system",
                "content": "The following is a voice message transcript. Only answer in JSON.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]

response_format={
            "type": "json_schema",
            "schema": VoiceNote.model_json_schema(),
        }

result = generate_with_multiple_input(messages, response_format = response_format)
result_json = json.loads(result['content'])
print(json.dumps(result_json, indent=2))