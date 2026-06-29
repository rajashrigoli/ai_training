import os

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

files = ["cape_town.txt", "tokyo.txt", "madrid.txt"]

for file_name in files:
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r") as f:
        journal_content = f.read()

    print(journal_content)

    prompt = f"""Please extract a comprehensive list of the restaurants 
    and their respective best dishes mentioned in the following journal entry. 
    Ensure that each restaurant name is accurately identified and listed. 

    Provide your answer in CSV format, ready to save. 
    Exclude the "```csv" declaration, don't add spaces after the comma, include column headers.

    Format:
    Restaurant, Dish
    Res_1, Dsh_1
    ...

    Journal entry:
    {journal_content}
    """

    restaurants_csv_ready_string = get_llm_response(prompt)

    print(restaurants_csv_ready_string)   