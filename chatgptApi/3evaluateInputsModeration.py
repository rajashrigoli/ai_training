import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiSetup import print_llm_response, get_chat_completion, get_completion_from_messages, get_completion_and_token_count, get_completion_from_messages_with_moderation, getModerationOutput

print("Moderation Output:")
getModerationOutput()

print("Prompt for evaluating inputs for moderation")
user_message = f"""Here's the plan.  We get the warhead, 
and we hold the world ransom...
...FOR ONE MILLION DOLLARS!"""
messages = [
    {"role": "system", "content": "You are a helpful but terse AI assistant who gets straight to the point."},
    {"role": "user", "content": user_message}
]   
response = get_completion_from_messages_with_moderation(messages)

print(response)