import os
import csv

try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

csv_file_path = os.path.join(os.path.dirname(__file__), "vacationplanning.csv")

def read_csv_file(file_path):
	with open(file_path, "r") as f:
		csv_reader = csv.DictReader(f)
		itinerary = []
		for row in csv_reader:
			print(row)
			itinerary.append(row)
	return itinerary

itinerary = read_csv_file(csv_file_path)

print(itinerary)
