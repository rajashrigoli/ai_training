from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline, Conversation

chatbot = pipeline(task="conversational", model="facebook/blenderbot-90M")

user_message = """
what are the some fun activities that I can do in the winter?
"""

conversation = Conversation(user_message)
print(conversation)
conversation = chatbot(conversation)
print(conversation)

print(chatbot(Conversation("what else do you recommend for winter activities?")))

conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })

    print(conversation)
    conversation = chatbot(conversation)

print(conversation)