from helper_functions import *

print("¯\_(ツ)_/¯")
print_llm_response("What is the capital of France?")
type(17)
print( len("Hello World!") ) 
print(round(42.17))
string_length = len("Hello World!")
print(string_length)

name = "Tommy"
potatoes = 4.75
prompt = f"""Write a couplet about my friend {name} who has about {round(potatoes)} potatoes"""
response = get_llm_response(prompt)
print(response)

lucky_number = 2 * 10
print(f"Your lucky number is {lucky_number}!")

# Use print_llm_response() to print a poem with the specified number of lines. Use the 
# prompt variable to save your prompt before calling print_llm_response()

number_of_lines = 4
prompt = f"Write a poem with {number_of_lines} lines."
print_llm_response(prompt)

# Repeat exercise 2, this time using the function get_llm_response(), then print() to print it. This function asks 
# the LLM for a response, just like print_llm_response, but does not print it. You'll need to save the response to
# a variable, then print it out separately.

number_of_lines = 6
prompt = f"Write a poem with {number_of_lines} lines."
response = get_llm_response(prompt)
print(response)