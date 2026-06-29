try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import os
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response


list_of_tasks = [
    "Compose a brief email to my boss explaining that I will be late for tomorrow's meeting.",
    "Write a birthday poem for Otto, celebrating his 28th birthday.",
    "Write a 300-word review of the movie 'The Arrival'."
]
print(list_of_tasks)

for task in list_of_tasks:
    print_llm_response(task)
print("******* Looping through a list of ice cream flavors *******")
ice_cream_flavors = [
    "Vanilla",
    "Chocolate",
    "Strawberry",
    "Mint Chocolate Chip"
]

for flavor in ice_cream_flavors:
    prompt = f"""For the ice cream flavor listed below, 
    provide a captivating description that could be used for promotional purposes.

    Flavor: {flavor}

    """
    print_llm_response(prompt)


print("******* Looping through a list of ice cream flavors and saving responses *******")
ice_cream_descriptions = []

for flavor in ice_cream_flavors:
    prompt = f"""For the ice cream flavor listed below, 
    provide a captivating description that could be used for promotional purposes.

    Flavor: {flavor}

    """
    response = get_llm_response(prompt)
    ice_cream_descriptions.append(response)

print("Ice Cream Descriptions:")
print(ice_cream_descriptions)

print("******* Looping through a list of words with typos and saving corrected words *******")
words_with_typos = ["Aple", "Wether", "Newpaper"]
words_without_typos = []

for word in words_with_typos:
    prompt = f"""Fix the spelling mistake in the following word: {word}
    Provide only the word.
    """
    correct_word = get_llm_response(prompt)
    words_without_typos.append(correct_word)

print(words_without_typos)