import os

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

cape_town_path = os.path.join(os.path.dirname(__file__), "cape_town.txt")

with open(cape_town_path, "r") as f:
	journal_cape_town = f.read()

print(journal_cape_town)

tokyo_path = os.path.join(os.path.dirname(__file__), "tokyo.txt")
with open(tokyo_path, "r") as f:
    journal_tokyo = f.read()

print(journal_tokyo)

madrid_path = os.path.join(os.path.dirname(__file__), "madrid.txt")
with open(madrid_path, "r") as f:
    journal_madrid = f.read()

print(journal_madrid)

prompt = f"""Respond with "Relevant" or "Not relevant": 
the journal describes restaurants and their specialties. 

Journal:
{journal_tokyo}"""

print(prompt)

print_llm_response(prompt)