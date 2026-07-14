import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_llm_response, get_chat_completion

allReviews = [
    'The mochi is excellent!',
    'Best soup dumplings I have ever eaten.',
    'Not worth the 3 month wait for a reservation.',
    'The colorful tablecloths made me smile!',
    'The pasta was cold.'
]
all_sentiments = []
print("Classify the following reviews as having either a positive or negative sentiment:\n")
for review in allReviews:
    prompt = f"""
    Classify the following review 
        as having either a positive or
        negative sentiment:

        {review}
    """
    sentiment = get_llm_response(prompt)
    all_sentiments.append(sentiment)
    print(f"Review: {review}\nSentiment: {sentiment}\n")


print("Count of positive and negative sentiments:")
print(all_sentiments)
positive_count = 0
negative_count = 0
for sentiment in all_sentiments:
    if sentiment == "Positive." :
        positive_count += 1
    elif sentiment == "Negative.":
        negative_count += 1
print(f"Positive: {positive_count}, Negative: {negative_count}")