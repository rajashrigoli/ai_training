import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_llm_response, get_chat_completion

print('Define a prompt that will classify the sentiment of a restaurant review.')

prompt = """
Classify the following review 
    as having either a positive or
    negative sentiment:

    The banana pudding was really tasty!
"""

print_llm_response(prompt)