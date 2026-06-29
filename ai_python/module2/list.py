try:
	from ai_python.helper_functions import print_llm_response
except ModuleNotFoundError:
	import os
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response

name = "Shivanshika"
prompt = f"""
Write a four line birthday poem for my friend {name}. 
The poem should be inspired by the first letter of my friend's name.
"""
print_llm_response(prompt)

friends_list = ["Shivanshika", "Aarav", "Ishaan", "Aarika", "Anaya"]
print(friends_list)
len(friends_list)
type(friends_list)

prompt = f"""
Write a set of four line birthday poems for my friends {friends_list}. 
The poems should be insipred by the first letter of each friend's name.
"""
print(prompt)

print_llm_response(prompt)

first_friend = friends_list[0]
print(first_friend)

print("******* Adding a new friend to the list *******")
friends_list.append("Anvita")
print(friends_list)

print("******* Removing a friend from the list *******")
friends_list.remove("Aarav")
print(friends_list)

print("********* List with other data types *********")
list_ages = [42, 28, 30]
print(list_ages)

list_of_tasks = [
    "Compose a brief email to my boss explaining that I will be late for tomorrow's meeting.",
    "Write a birthday poem for Otto, celebrating his 28th birthday.",
    "Write a 300-word review of the movie 'The Arrival'."
]

task = list_of_tasks[0]
print_llm_response(task)

task = list_of_tasks[1]
print_llm_response(task)

task = list_of_tasks[2]
print_llm_response(task)