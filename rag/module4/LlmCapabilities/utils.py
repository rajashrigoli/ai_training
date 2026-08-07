import json
import requests
import os
import re
from typing import List, Dict, Any
from together import Together


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

    groq_api_key = os.environ.get('GROQ_API_KEY', '')
    
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                            json=payload, headers=headers)
    if not response.ok:
        raise Exception(f"Error while calling LLM: {response.text}")
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
    groq_api_key = os.environ.get('GROQ_API_KEY', '')
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                            json=payload, headers=headers)
    if not response.ok:
        raise Exception(f"Error while calling LLM: {response.text}")
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


def print_response(response):
    """
    Prints a formatted chatbot response with color-coded roles.

    The function uses ANSI escape codes to apply text styles. Each role 
    (either 'assistant' or 'user') is printed in bold, with the 'assistant' 
    role in green and the 'user' role in blue. The content of the response 
    follows the role name.

    Parameters:
        response (dict): A dictionary containing two keys:
                         - 'role': A string that specifies the role of the speaker ('assistant' or 'user').
                         - 'content': A string with the message content to be printed.
    """
    # ANSI escape codes
    BOLD = "\033[1m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    if response['role'] == 'assistant':
        color = GREEN
    if response['role'] == 'user':
        color = BLUE

    s = f"{BOLD}{color}{response['role'].capitalize()}{RESET}: {response['content']}"
    print(s)


def chat(temperature = None, 
         top_k = None, 
         top_p = None,
         repetition_penalty = None,
         context = None):
    """
    Runs an interactive chat session between the user and an AI assistant.

    The chat continues in a loop until the user types 'STOP'. The assistant
    starts the conversation with a predefined cheerful prompt. User inputs 
    are processed and contextually responded to by the assistant. Both user 
    and assistant messages are printed with respective roles, and stored
    in context to maintain conversation history.

    Usage:
        Run the function and type your prompts. Type 'STOP' to end the chat.
    """
    if context is None:
        context = [{"role": "assistant", "content": "Hey there! How can I help you today?"}]
    
    # Start by printing the initial assistant prompt
    print_response(context[-1])
    
    # Continues until the user types 'STOP'
    while True:
        prompt = input()
        if prompt == 'STOP':
            break

        # Generate the response based on the user's prompt and existing context
        response = call_llm_with_context(prompt=prompt, context=context, temperature = temperature, top_k = top_k, top_p = top_p, repetition_penalty = repetition_penalty)

        # Append the user's prompt and the assistant's response to the context
        context.append({"role": "user", "content": prompt})
        context.append(response)

        # Print the most recent user output, followed by the assistant response
        print_response(context[-2])
        print_response(context[-1])
