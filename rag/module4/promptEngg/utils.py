import json
import requests
import os
import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env'))


def _extract_retry_after_seconds(response_text: str, default_wait: float = 6.0) -> float:
    """Extract wait time from Groq rate-limit message like 'Please try again in 5.66s'."""
    match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", response_text, flags=re.IGNORECASE)
    if not match:
        return default_wait
    try:
        return max(float(match.group(1)), 0.5)
    except ValueError:
        return default_wait


def _post_with_rate_limit_retry(url: str, payload: Dict[str, Any], headers: Dict[str, str], max_retries: int = 3):
    """Post to Groq API and retry briefly when TPM rate limits are hit."""
    for attempt in range(max_retries + 1):
        response = requests.post(url, json=payload, headers=headers)
        if response.ok:
            return response

        response_text = response.text
        is_rate_limit = response.status_code == 429 or "rate_limit_exceeded" in response_text.lower()
        if is_rate_limit and attempt < max_retries:
            wait_seconds = _extract_retry_after_seconds(response_text)
            time.sleep(wait_seconds)
            continue

        raise Exception(f"Error while calling LLM: {response_text}")

    raise Exception("Error while calling LLM: exhausted retry attempts.")


def get_proxy_url():
    """
    Get the proxy URL from environment variable or fall back to Together.ai endpoint.
    Uses TOGETHER_BASE_URL environment variable set in Dockerfile.
    Defaults to https://api.together.xyz/ if not set.
    """
    return os.environ.get('TOGETHER_BASE_URL', 'https://api.together.xyz/')

def get_proxy_headers():
    """
    Get the appropriate headers for API calls based on the platform.
    Returns Authorization header with Together API key if available.
    """
    return {"Authorization": os.environ.get("TOGETHER_API_KEY", "")}

def get_together_key():
    """
    Get the Together API key from environment variables.
    """
    return os.environ.get("TOGETHER_API_KEY", "")

def generate_params_dict(
    prompt: str,
    temperature: float = None,
    role = 'user',
    top_p: float = None,
    max_tokens: int = 500,
    model: str = "llama-3.1-8b-instant"
):
    """
    Call an LLM with different sampling parameters to observe their effects.

    Args:
        prompt: The text prompt to send to the model
        temperature: Controls randomness (lower = more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        model: The model to use

    Returns:
        The LLM response
    """

    # Create the dictionary with the necessary parameters
    kwargs = {"prompt": prompt, 'role':role, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, 'model': model}


    return kwargs

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="llama-3.1-8b-instant",
                               together_api_key = None,
                              **kwargs):
    groq_api_key = together_api_key or os.environ.get('GROQ_API_KEY', '')
    if not groq_api_key:
        raise Exception("Missing GROQ_API_KEY. Set it in environment or in the project .env file.")

    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
        **kwargs
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    response = _post_with_rate_limit_retry(
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        headers,
    )
    try:
        json_dict = json.loads(response.text)
    except Exception as e:
        raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict


def generate_with_multiple_input(messages: List[Dict],
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str ="llama-3.1-8b-instant",
                                together_api_key = None,
                                **kwargs):
    groq_api_key = together_api_key or os.environ.get('GROQ_API_KEY', '')
    if not groq_api_key:
        raise Exception("Missing GROQ_API_KEY. Set it in environment or in the project .env file.")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    response = _post_with_rate_limit_retry(
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        headers,
    )
    try:
        json_dict = json.loads(response.text)
    except Exception as e:
        raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict

def call_llm_with_context(prompt: str, context: list,  role: str = 'user', **kwargs):
    """
    Calls a language model with the given prompt and context to generate a response.

    Parameters:
    - prompt (str): The input text prompt provided by the user.
    - role (str): The role of the participant in the conversation, e.g., "user" or "assistant".
    - context (list): A list representing the conversation history, to which the new input is added.
    - **kwargs: Additional keyword arguments for configuring the language model call (e.g., top_k, temperature).

    Returns:
    - response (str): The generated response from the language model based on the provided prompt and context.
    """

    # Append the dictionary {'role': role, 'content': prompt} into the context list
    context.append({'role': role, 'content': prompt})

    # Call the llm with multiple input passing the context list and the **kwargs
    response = generate_with_multiple_input(context, **kwargs)

    # Append the LLM response in the context dict
    context.append(response)

    return response


def check_if_outfit_or_supplement(query):
    prompt = f"""
Determine the category of the following query as either "nutritional" or "outfit" related.
- Nutritional queries: These are related to nutrition products, such as whey protein, vitamins, supplements, dietary products, and health-related food and beverages.
  - Outfit queries: These pertain to clothing and fashion, including items like shirts, dresses, shoes, accessories, and jewelry.
Examples:

1. Query: “Where can I buy high-protein snacks?” Expected answer: Nutritional
2. Query: “Best shirt styles for summer 2023” Expected answer: Outfit
3. Query: “Are there any shoes designed for running?” Expected answer: Outfit
4. Query: “What multivitamins should I take daily?” Expected answer: Nutritional
5. Query: “Best weight loss products that are stylish” Expected answer: Nutritional
6. Query: “Athletic wear that boosts performance” Expected answer: Outfit 

Query: {query}

Instructions: Respond with “Nutritional” if the query pertains to nutritional products or “Outfit” if it pertains to clothing or fashion products.
Answer only one single word.
"""
    return prompt
    

def decide_if_technical_or_creative(query):
    """
    Determines whether a given query is creative or technical in nature.

    Args:
        query (str): The query string to be evaluated.

    Returns:
        str: A label indicating the query type, either 'creative' or 'technical'.

    This function constructs a prompt to classify a query based on its content. 
    Creative queries typically involve requests to generate original content, whereas 
    technical queries relate to documentation or technical information, such as procedures.
    By leveraging an LLM, it identifies the query type and returns an appropriate label.
    """
    
    PROMPT = f"""Decide if the following query is a creative query or a technical query.
    Creative queries ask you to create content, while technical queries are related to documentation or technical requests, like information about procedures.
    Answer only 'creative' or 'technical'.
    Query: {query}
    """
    result = generate_with_single_input(PROMPT)
    label = result['content']
    return label

def answer_query(query):
    """
    Processes a query and generates an appropriate response by categorizing the query
    as either 'technical' or 'creative', and modifies behavior based on this categorization.

    Args:
        query (str): The query string to be answered.

    Returns:
        str: A generated response from the LLM tailored to the nature of the query.

    This function first determines the nature of the query using the `decide_if_technical_or_creative` function. 
    If the query is classified as 'technical', it sets parameters suitable for precise and low-variability responses. 
    If the query is 'creative', it applies parameters allowing for more variability and creativity. 
    If the classification is inconclusive, it uses neutral parameters. 
    It then generates a response using these parameters and returns the content.
    """
    
    # Determine whether the query is 'technical' or 'creative'
    label = decide_if_technical_or_creative(query).lower()

    # Set parameters for technical queries (precise, low randomness)
    if label == 'technical':
        kwargs = generate_params_dict(query, temperature=0, top_p=0.1)
    
    # Set parameters for creative queries (variable, high randomness)
    elif label == 'creative':
        kwargs = generate_params_dict(query, temperature=1.1, top_p=0.4)

    # Use default parameters if the query type is inconclusive
    else:
        kwargs = generate_params_dict(query, temperature=0.5, top_p=0.5)
    
    # Generate a response based on the query type and parameters
    response = generate_with_single_input(**kwargs)
    
    # Extract and return the content from the response
    result = response['content']
    return result

def generate_system_call(command):
    PROMPT = f"""
You are an assistant program that converts natural language commands into structured JSON for controlling smart home devices. The JSON should conform to a specific format describing the device, action, and parameters. Here's how you can do it:

**Available Devices and Actions:**

1. **Light**
   - Actions: "turn on", "turn off"
   - Parameters: color, intensity (percentage)

2. **Automatic Lock**
   - Actions: "lock", "unlock"
   - Parameters: None

3. **Sound System (Speaker)**
   - Actions: "play", "pause", "stop", "set volume"
   - Parameters: volume (integer), track (string), playlist_style (string)

4. **TV**
   - Actions: "turn on", "turn off", "change channel", "adjust volume"
   - Parameters: channel (string), volume (integer)

5. **Air Conditioner**
   - Actions: "turn on", "turn off", "set temperature", "adjust fan speed"
   - Parameters: temperature (integer), fan_speed (low/medium/high)

**Rooms and Devices:**
- **Office**
  - Lights: "office_light_1" (ID: 123), "office_light_2" (ID: 321)
  - Automatic Lock: "office_door_lock" (ID: 111)

- **Living Room**
  - Light: "living_room_light" (ID: 222)
  - Speaker: "living_room_speaker" (ID: 223)
  - Air Conditioner: "living_room_airconditioner" (ID: 556)

- **Kitchen**
  - Light: "kitchen_light" (ID: 333)

- **Bedroom**
  - Light: "bedroom_light" (ID: 444)
  - TV: "bedroom_tv" (ID: 445)

- **Bathroom**
  - Light: "bathroom_light" (ID: 555)

**Task:**
Convert the following natural language command into the structured JSON format based on the available devices:

**Input Examples:**

1. "Turn on the office light with ID 123 with blue color and 50% intensity."
   - JSON:
     [
     {{
       "room": "office",
       "object_id": "123",
       "object_name": "office_light_1",
       "action": "turn on",
       "parameters": {{"color": "blue", "intensity": "50%"}}
     }}
     ]

2. "Lock the office door."
   - JSON:
   [
     {{
       "room": "office",
       "object_id": "111",
       "object_name": "office_door_lock",
       "action": "lock",
       "parameters": {{}}
     }}
    ]

2. "Make my living room a cheerful place"
   - JSON:
   [
     {{
       "room": "living_room",
       "object_id": "222",
       "object_name": "living_room_light",
       "action": "turn on",
       "parameters": {{'intensity': '80%', 'color':'yellow'}}
     }},
     {{
       "room": "living_room",
       "object_id": "223",
       "object_name": "living_room_speaker",
       "action": "turn on",
       "parameters": {{'volume': '100', 'playlist_style':'party'}}
     }},
     
   ]

**Note:**
- Ensure that each JSON object correctly maps the natural command to the appropriate device and action using the listed device ID.
- Use the object ID to differentiate between devices when the room contains multiple similar items.
- You can add more than one parameter in the parameters dictionary.

Using this information, translate the following command into JSON: "{command}". Output a list with all the necessary JSONs. 
Always output a list even if there is only one command to be applied, do not output anything else but the desired structure.
"""
    kwargs = generate_params_dict(PROMPT, temperature=0.4, top_p=0.1, max_tokens=220)
    result = generate_with_single_input(**kwargs)
    return result['content']