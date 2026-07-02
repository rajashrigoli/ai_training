# import gradio as gr
import os

from openai import OpenAI
from dotenv import load_dotenv

import random

#Get the OpenAI API key from the .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '', '.env'), override=True)

# Set up the OpenAI client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

AI_MODEL = "llama-3.3-70b-versatile"


def print_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then prints the response of the model.
    """
    llm_response = get_llm_response(prompt)
    print(llm_response)


def get_llm_response(prompt):
    """This function takes as input a prompt, which must be a string enclosed in quotation marks,
    and passes it to OpenAI's GPT3.5 model. The function then saves the response of the model as
    a string.
    """
    try:
        if not isinstance(prompt, str):
            raise ValueError("Input must be a string enclosed in quotes.")
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful but terse AI assistant who gets straight to the point.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        return response
    except TypeError as e:
        print("Error:", str(e))


def get_chat_completion(prompt, history):
    history_string = "\n\n".join(["\n".join(turn) for turn in history])
    prompt_with_history = f"{history_string}\n\n{prompt}"
    completion = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful but terse AI assistant who gets straight to the point.",
            },
            {"role": "user", "content": prompt_with_history},
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content
    return response


# def open_chatbot():
#     """This function opens a Gradio chatbot window that is connected to OpenAI's GPT3.5 model."""
#     gr.close_all()
#     gr.ChatInterface(fn=get_chat_completion).launch(quiet=True)

def get_dog_age(human_age):
    """This function takes one parameter: a person's age as an integer and returns their age if
    they were a dog, which is their age divided by 7. """
    return human_age / 7

def get_goldfish_age(human_age):
    """This function takes one parameter: a person's age as an integer and returns their age if
    they were a dog, which is their age divided by 5. """
    return human_age / 5

def get_cat_age(human_age):
    if human_age <= 14:
        # For the first 14 human years, we consider the age as if it's within the first two cat years.
        cat_age = human_age / 7
    else:
        # For human ages beyond 14 years:
        cat_age = 2 + (human_age - 14) / 4
    return cat_age

def get_random_ingredient():
    """
    Returns a random ingredient from a list of 20 smoothie ingredients.
    
    The ingredients are a bit wacky but not gross, making for an interesting smoothie combination.
    
    Returns:
        str: A randomly selected smoothie ingredient.
    """
    ingredients = [
        "rainbow kale", "glitter berries", "unicorn tears", "coconut", "starlight honey",
        "lunar lemon", "blueberries", "mermaid mint", "dragon fruit", "pixie dust",
        "butterfly pea flower", "phoenix feather", "chocolate protein powder", "grapes", "hot peppers",
        "fairy floss", "avocado", "wizard's beard", "pineapple", "rosemary"
    ]
    
    return random.choice(ingredients)

def get_random_number(x, y):
    """
        Returns a random integer between x and y, inclusive.
        
        Args:
            x (int): The lower bound (inclusive) of the random number range.
            y (int): The upper bound (inclusive) of the random number range.
        
        Returns:
            int: A randomly generated integer between x and y, inclusive.

        """
    return random.randint(x, y)

def calculate_llm_cost(characters, price_per_1000_tokens=0.015):
    tokens = characters / 4
    cost = (tokens / 1000) * price_per_1000_tokens
    return f"${cost:.4f}"

def display_map():
    # Define the bounding box for the continental US
    us_bounds = [[24.396308, -125.0], [49.384358, -66.93457]]
    # Create the map centered on the US with limited zoom levels
    m = folium.Map(
	    location=[37.0902, -95.7129],  # Center the map on the geographic center of the US
	    zoom_start=5,  # Starting zoom level
	    min_zoom=4,  # Minimum zoom level
	    max_zoom=10,
	    max_bounds=True,
	    control_scale=True  # Maximum zoom level
	)

    # Set the bounds to limit the map to the continental US
    m.fit_bounds(us_bounds)
    # Add a click event to capture the coordinates
    m.add_child(folium.LatLngPopup())
    title_html = '''
	<div style="
	display: flex;
	justify-content: center;
	align-items: center;
	width: 100%; 
	height: 50px; 
	border:0px solid grey; 
	z-index:9999; 
	font-size:30px;
	padding: 5px;
	background-color:white;
	text-align: center;
	">
	&nbsp;<b>Click to view coordinates</b>
	</div>
	'''
	
    m.get_root().html.add_child(folium.Element(title_html))

    # Display the map
    return m
    
def get_forecast(lat, lon):
    url = f"https://api.weather.gov/points/{lat},{lon}"

    # Make the request to get the grid points
    response = requests.get(url)
    data = response.json()
    # Extract the forecast URL from the response
    forecast_url = data['properties']['forecast']

    # Make a request to the forecast URL for the selected location
    forecast_response = requests.get(forecast_url)
    forecast_data = forecast_response.json()
    
    daily_forecast = forecast_data['properties']['periods']
    return daily_forecast