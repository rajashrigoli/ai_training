import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_chat_completion, get_completion_from_messages, get_completion_and_token_count   

delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category. 
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human

"""
dash_line = '-' *  100
print("Assigning categories to customer service queries")
user_message = f"""
I want you to delete my profile and all of my user data"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"}
]

response = get_completion_from_messages(messages, temperature=0.0)
print(dash_line)
print(user_message)
print(dash_line)
print(response)


user_message = f"""I want to know how to update my payment method"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"}
]
response = get_completion_from_messages(messages, temperature=0.0)
print(dash_line)
print(user_message)
print(dash_line)
print(response)


user_message = f"""I want to know if my device is compatible with your software"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"}
]
response = get_completion_from_messages(messages, temperature=0.0)
print(dash_line)
print(user_message)
print(dash_line)
print(response)

user_message = f"""Tell me more about your flat screen tvs"""
messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"}
]
response = get_completion_from_messages(messages, temperature=0.0)
print(dash_line)
print(user_message)
print(dash_line)    
