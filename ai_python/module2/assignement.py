try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import os
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

print("******** ## Exercise 1: The Bookworm's Inventory   ********")
book = {
    "title": "To Kill a Mockingbird",
    "author": "Harper Lee",
    "on_shelf": "False",
    "borrower": "Arthur Dent",
    "overdue": "True",
    "on_hold": "False"
}
print(book)
print("******** ## Exercise 2: Is It On the Shelf?   ********")
if(book["on_shelf"] == "True" and book["on_hold"] == "False"):
    print("Book is available to be borrowed.")
else:
    print("Book is not available to be borrowed.")

print("******** ## Exercise 3: On Hold or Overdue?  ********")
if(book["overdue"] == "True"):
    print(f"Book is overdue. Contact {book['borrower']} to return the book.")
else:
    book['on_hold'] = "True"
    print("Book has been put on hold")

print("******** ## Exercise 4: Tracking Down the Borrower  ********")
# List of dictionaries with borrower contact information
borrowers_list = [
    {
        "name": "Alice Johnson",
        "email": "alice.johnson@dlailibrary.com",
        "phone": "+1111111111"
    },
    {
        "name": "Bob Smith",
        "email": "bob.smith@dlailibrary.com",
        "phone": "+2222222222"
    },
    {
        "name": "Arthur Dent",
        "email": "arthur.dent@dlailibrary.com",
        "phone": "+3333333333"
    },
    {
        "name": "Diana Prince",
        "email": "diana.prince@dlailibrary.com",
        "phone": "+4444444444"
    }
]

borrower_email = None

for borrower in borrowers_list:
    if(borrower['name'] == book['borrower']):
        borrower_email = borrower['email']
        break

if borrower_email:
    print(f"{book['borrower']}'s email is: {borrower_email}")

print("******** ## Exercise 5: The LLM to the Rescue! ********")

# Name of the borrower
person_name = book['borrower']

# Name of the book
book_name = book['title']

# Name of book's author
book_author = book['author']

# Due Date
due_date = "16 November 2024"

propmt = f"""Please write a polite email to {person_name} reminding them that the book '{book_name}' by {book_author} is overdue. 
The due date was {due_date}. 
Please ask them to return the book as soon as possible. Thank you!"""

print_llm_response(propmt)