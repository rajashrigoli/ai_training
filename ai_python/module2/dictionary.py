try:
	from ai_python.helper_functions import print_llm_response, get_llm_response
except ModuleNotFoundError:
	import os
	import sys

	sys.path.append(os.path.dirname(os.path.dirname(__file__)))
	from helper_functions import print_llm_response, get_llm_response

ice_cream_flavors = [
    "Vanilla: Classic and creamy with a rich, smooth flavor from real vanilla beans.",
    "Chocolate: Deep and indulgent, made with rich cocoa for a satisfying chocolate experience.",
    "Strawberry: Sweet and fruity, bursting with the fresh taste of ripe strawberries.",
    "Mint Chocolate Chip: Refreshing mint ice cream studded with decadent chocolate chips.",
    "Cookie Dough: Vanilla ice cream loaded with chunks of chocolate chip cookie dough.",
    "Salted Caramel: Sweet and salty with a smooth caramel swirl and a hint of sea salt.",
    "Pistachio: Nutty and creamy, featuring the distinct taste of real pistachios.",
    "Cookies and Cream: Vanilla ice cream packed with chunks of chocolate sandwich cookies.",
    "Mango: Tropical and tangy, made with juicy mangoes for a refreshing treat.",
    "Rocky Road: Chocolate ice cream mixed with marshmallows, nuts, and chocolate chunks."
]
print(ice_cream_flavors[0])

ice_cream_flavors = {
    "Vanilla": "Classic and creamy with a rich, smooth flavor from real vanilla beans.",
    "Chocolate": "Deep and indulgent, made with rich cocoa for a satisfying chocolate experience.",
    "Strawberry": "Sweet and fruity, bursting with the fresh taste of ripe strawberries.",
    "Mint Chocolate Chip": "Refreshing mint ice cream studded with decadent chocolate chips.",
    "Cookie Dough": "Vanilla ice cream loaded with chunks of chocolate chip cookie dough.",
    "Salted Caramel": "Sweet and salty with a smooth caramel swirl and a hint of sea salt.",
    "Pistachio": "Nutty and creamy, featuring the distinct taste of real pistachios.",
    "Cookies and Cream": "Vanilla ice cream packed with chunks of chocolate sandwich cookies.",
    "Mango": "Tropical and tangy, made with juicy mangoes for a refreshing treat.",
    "Rocky Road": "Chocolate ice cream mixed with marshmallows, nuts, and chocolate chunks."
}


print("******** List of Ice Cream Flavors ********")
print(ice_cream_flavors.keys())

print("******** List of Ice Cream Flavors with Descriptions ********")
print(ice_cream_flavors.values())

cookie_dough_description = ice_cream_flavors["Cookie Dough"]
print(cookie_dough_description)

print(ice_cream_flavors)

ice_cream_flavors["Rocky Road"] = "Chocolate ice cream mixd witother ngredients."
print(ice_cream_flavors)


list_of_tasks = [
    "Compose a brief email to my boss explaining that I will be late for tomorrow's meeting.",
    "Write a birthday poem for Otto, celebrating his 28th birthday.",
    "Write a 300-word review of the movie 'The Arrival'.",
    "Draft a thank-you note for my neighbor Dapinder who helped water my plants while I was on vacation.",
    "Create an outline for a presentation on the benefits of remote work."
]

#instead of that unorganized large list, divide tasks by priority
high_priority_tasks = [
    "Compose a brief email to my boss explaining that I will be late for tomorrow's meeting.",
    "Create an outline for a presentation on the benefits of remote work."
]

medium_priority_tasks = [
    "Write a birthday poem for Otto, celebrating his 28th birthday.",
    "Draft a thank-you note for my neighbor Dapinder who helped water my plants while I was on vacation."
]

low_priority_tasks = [
    "Write a 300-word review of the movie 'The Arrival'."
]

#create dictionary with all tasks
#dictionaries can contain lists!
prioritized_tasks = {
    "high_priority": high_priority_tasks,
    "medium_priority": medium_priority_tasks,
    "low_priority": low_priority_tasks
}
print("******** Prioritized Tasks Dictionary ********")
print(prioritized_tasks)
print("")
print(prioritized_tasks["high_priority"])
#complete high priority tasks 
for task in prioritized_tasks["high_priority"]:
    print_llm_response(task)

print("******* Food Preferences Dictionary Example *******")
food_preferences_tommy = {
        "dietary_restrictions": "vegetarian",
        "favorite_ingredients": ["tofu", "olives"],
        "experience_level": "intermediate",
        "maximum_spice_level": 6
}

prompt = f"""Please suggest a recipe that tries to include 
the following ingredients: 
{food_preferences_tommy["favorite_ingredients"]}.
The recipe should adhere to the following dietary restrictions:
{food_preferences_tommy["dietary_restrictions"]}.
The difficulty of the recipe should be: 
{food_preferences_tommy["experience_level"]}
The maximum spice level on a scale of 10 should be: 
{food_preferences_tommy["maximum_spice_level"]} 
Provide a two sentence description.
"""

print_llm_response(prompt)

available_spices = ["cumin", "turmeric", "oregano", "paprika"]
prompt = f"""Please suggest a recipe that tries to include 
the following ingredients: 
{food_preferences_tommy["favorite_ingredients"]}.
The recipe should adhere to the following dietary restrictions:
{food_preferences_tommy["dietary_restrictions"]}.
The difficulty of the recipe should be: 
{food_preferences_tommy["experience_level"]}
The maximum spice level on a scale of 10 should be: 
{food_preferences_tommy["maximum_spice_level"]} 
Provide a two sentence description.

The recipe should not include spices outside of this list:
Spices: {available_spices}
"""
print(prompt)
recipe = get_llm_response(prompt)
print(recipe)