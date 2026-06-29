import os

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

email_path = os.path.join(os.path.dirname(__file__), "email.txt")
with open(email_path, "r") as f:
	email = f.read()

print(email)

recipe_path = os.path.join(os.path.dirname(__file__), "recipe.txt")
with open(recipe_path, "r") as f:
	recipe = f.read()

print(recipe)