from math import cos, sin, pi, floor
from random import sample

import os

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

print(pi)
float_num = 3.14159
print(floor(float_num))

spices = ["cumin", "turmeric", "oregano", "paprika"]
vegetables = ["lettuce", "tomato", "carrot", "broccoli"]
proteins = ["chicken", "tofu", "beef", "fish", "tempeh"]

random_spices = sample(spices, 2)
random_vegetables = sample(vegetables, 2)
random_proteins = sample(proteins, 1)

prompt = f"""Please suggest a recipe that tries to include the following ingredients: 
{random_vegetables + random_proteins + random_spices}.
The recipe should adhere to the following dietary restrictions: vegetarian.
The difficulty of the recipe should be: intermediate.
The maximum spice level on a scale of 10 should be: 6.
Provide a two sentence description.
The recipe should not include spices outside of this list:
Spices: {spices}
"""

print_llm_response(prompt)