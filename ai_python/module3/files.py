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

prompt = f"""Extract bullet points from the following email. 
Include the sender information. 

Email:
{email}"""

print(prompt)

bullet_points = get_llm_response(prompt)
print(bullet_points)