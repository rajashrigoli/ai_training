import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_chat_completion, get_completion_from_messages, get_completion_and_token_count

equal_line = "=" * 100
dash_line = "-" * 100
print(equal_line)
print("Prompt the model and get a completion")
print(equal_line)

print("What is the capital of France?")
print_llm_response("What is the capital of France?")

print("\n\n")
print(dash_line)
print("Tokens")
print(dash_line)

response = get_chat_completion("Take the letters in lollipop \
and reverse them")
print(response)

response = get_chat_completion("""Take the letters in \
l-o-l-l-i-p-o-p and reverse them""")
print(response)

print("\n\n")
print(dash_line)
print("As style of Dr Seuss")
print(dash_line)
messages =  [  
{'role':'system', 
 'content':"""You are an assistant who\
 responds in the style of Dr Seuss."""},    
{'role':'user', 
 'content':"""write me a very short poem\
 about a happy carrot"""},  
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)

# length
print("\n\n")
print(dash_line)
print("As one sentence")
print(dash_line)
messages =  [  
{'role':'system', 'content':'All your responses must be \
one sentence long.'},    
{'role':'user', 'content':'write me a story about a happy carrot'},  
] 
response = get_completion_from_messages(messages, temperature =1)
print(response)

# combined
print("\n\n")
print(dash_line)
print("As style of Dr Seuss and one sentence")
print(dash_line)
messages =  [  
{'role':'system',
 'content':"""You are an assistant who \
responds in the style of Dr Seuss. \
All your responses must be one sentence long."""},    
{'role':'user', 'content':"""write me a story about a happy carrot"""},
] 
response = get_completion_from_messages(messages, temperature =1)
print(response)

print("\n\n")
print(dash_line)
print("Get completion and token count")
print(dash_line)

messages = [
{'role':'system', 
 'content':"""You are an assistant who responds\
 in the style of Dr Seuss."""},    
{'role':'user', 'content':"""write me a very short poem \
 about a happy carrot"""},  
] 
response, token_dict = get_completion_and_token_count(messages)
print("\nResponse:")
print(response)
print("\nToken Count")
print(token_dict)