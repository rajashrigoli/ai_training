#!pip3 install bs4


from bs4 import BeautifulSoup

import requests
import os

try:
	from IPython.display import HTML, display
except Exception:
	HTML = None
	display = None

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response


# The url from one of the Batch's newsletter
url = 'https://www.deeplearning.ai/the-batch/the-world-needs-more-intelligence/'

# Getting the content from the webpage's contents
response = requests.get(url)

# Print the response from the requests
print(response)

if HTML and display:
	display(HTML(f'<iframe src="{url}" width="60%" height="400"></iframe>'))
else:
	print(f"Open in browser: {url}")

# Using beautifulsoup to extract the text
soup = BeautifulSoup(response.text, 'html.parser')
# Find all the text in paragraph elements on the webpage
all_text = soup.find_all('p')

# Create an empty string to store the extracted text
combined_text = ""

# Iterate over 'all_text' and add to the combined_text string
for text in all_text:
    combined_text = combined_text + "\n" + text.get_text()

# Print the final combined text
print(combined_text)

prompt = f"""Extract the key bullet points from the following text.

Text:
{combined_text}
"""

print_llm_response(prompt)