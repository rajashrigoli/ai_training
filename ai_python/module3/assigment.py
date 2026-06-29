import os

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

topics_to_use = [
    
    ### START CODE HERE ###
    
    {
        # Use your the "first" entry from "key_topics" as "Topic 1"
        # Hint: Remember, in Python, counting starts from zero (0)
        "Topic 1": "Gen AI",
        
        # Use a boolean value (True or False) for "to_use"
        "to_use": True
    },
    {
        # Use your the "second" entry from "key_topics" as "Topic 2"
        "Topic 2": "AI in QA",
        
        # Use a boolean value (True or False) for "to_use"
        "to_use": False
    },
    {
        # Use your the "third" entry from "key_topics" as "Topic 3"
        "Topic 3": "Python in AI",
        
        # Use a boolean value (True or False) for "to_use"
        "to_use": True
    }
]
    
    ### END CODE HERE ###

    #Write a prompt that asks the LLM to generate a poem. The poem should be of exactly 4 (four) lines (line). Your prompt should also include the topics_to_use list.

#You have the flexibility to structure your prompt and wording as you see fit. The key is to include all three pieces of information (`topics_to_use` list, mention of writing a "poem" and using only "4 (four) lines (line)") naturally within the `prompt`'s instructions. 

#For example, your prompt could look something like this: 
#"Using only the topics from the list <topics_to_use>, write a 4-line poem."

prompt = f"""Using only the topics from the list {topics_to_use}, write a poem that consists of exactly 4 (four) lines.
"""
print_llm_response(prompt)