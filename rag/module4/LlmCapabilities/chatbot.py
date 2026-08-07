import os
import json
import requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))

from utils import chat, print_response, call_llm_with_context


dash_line = "-" * 100

print(dash_line)
print("Simple Chatbot")
print(dash_line)


# Setting up a list to serve as the context. It will contain a system prompt and an initial assistant prompt.
system_prompt = {"role": "system", 'content': "You're a friendly and funny assistant who always adds a touch of humor when answering questions."}
assistant_prompt = {"role": "assistant", "content": "Hey there, fabulous! Ready to have some fun and get things done? How can this charming assistant help you today?"}
context = [system_prompt, assistant_prompt]


# To run again with different parameters, either write STOP or click the stop button in the Jupyter Lab panel
chat(context=context)
