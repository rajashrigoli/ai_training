import json
from weaviate.classes.query import Filter
import weaviate
import joblib

import json
from weaviate.classes.query import Filter
import weaviate
import joblib

import os
import logging
import warnings

# Disable HF specific noise
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HOME"] = "/tmp/huggingface_cache"

# Silence Loggers
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# Silence standard warnings
warnings.filterwarnings("ignore")

import unittests
from utils import (
    ChatWidget,
    generate_with_single_input,
    kill_processes_on_ports,
    generate_params_dict
)

# Kill processes on ports before importing flask_app and weaviate_server
# WARNING: Running this cell twice may kill the active kernel
kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

import flask_app
import weaviate_server


client = weaviate.connect_to_local(port=8079, grpc_port=50050)

# An output example is
kwargs = generate_params_dict("Solve x^2 - 1 = 0", temperature = 1.2, top_p = 0.2)
print(kwargs)

# Generating 
response = generate_with_single_input(**kwargs)
print(response['content'])

# Loading products data
PRODUCTS_DATA = joblib.load('dataset/clothes_json.joblib')

# Let's get one example
print(PRODUCTS_DATA[0])

FAQ = joblib.load("dataset/faq.joblib")

# Get an example
print(FAQ[:2])


# GRADED CELL 

def check_if_faq_or_product(query: str) -> str:
    """
    Determines whether a given instruction prompt is related to a frequently asked question (FAQ) or a product inquiry.

    Parameters:
    - query (str): The instruction or query to be labeled as either FAQ or product-related.

    Returns:
    - str: The label 'FAQ' if the prompt is classified as a frequently asked question, 'Product' if it relates to product information, or
      None if the label is inconclusive.
    """
    ### START CODE HERE ###

    # Set the hardcoded prompt. Remember to include the query, clear instructions (explicitly tell the LLM to return FAQ or Product)
    # Include examples of question / desired label pairs.

    prompt = (
        "Classify the following query as either 'FAQ' or 'Product'. "
        "Return ONLY the word 'FAQ' or 'Product'.\n\n"
        "Examples:\n"
        "Query: What is your return policy? Label: FAQ\n"
        "Query: Do you have blue T-shirts? Label: Product\n"
        "Query: How can I track my order? Label: FAQ\n"
        "Query: Show me red dresses under $50. Label: Product\n"
        "Query: Create a summer look for me. Label: Product\n\n"
        f"Query: {query} Label:"
    )

    # Get the kwargs dictionary to call the LLM, with PROMPT as prompt, low temperature (0.3 - 0.5)
    # The function call is generate_params_dict, pass the PROMPT and the correct temperature
    kwargs = generate_params_dict(prompt, temperature=0.3)

    # Call generate_with_single_input with **kwargs
    response = generate_with_single_input(**kwargs)
    # Get the label by accessing the 'content' key of the response dictionary

    label = response['content'].strip()

    ### END CODE HERE ###
    
    return label


queries = ['What is your return policy?', 
           'Give me three examples of blue T-shirts you have available.', 
           'How can I contact the user support?', 
           'Do you have blue Dresses?',
           'Create a look suitable for a wedding party happening during dawn.']

for query in queries:
    response = check_if_faq_or_product(query)
    label = response
    print(f"Query: {query} Label: {label}")

unittests.test_check_if_faq_or_product(check_if_faq_or_product)

# print the structure of the first element
print(FAQ[0])

def generate_faq_layout(faq_dict: list) -> str:
    """
    Generates a formatted string layout for a list of FAQs.

    This function iterates through a dictionary of frequently asked questions (FAQs) and constructs
    a string where each question is followed by its corresponding answer and type.

    Parameters:
    - faq_dict (list): A list of dictionaries, each containing keys 'question', 'answer', and 'type' 
      representing an FAQ entry.

    Returns:
    - str: A string representing the formatted layout of FAQs, with each entry on a separate line.
    """
    # Initialize an empty string
    t = ""

    # Iterate over every FAQ question in the FAQ list
    for f in faq_dict:
        # Append the question with formatted string (remember to use f-string and access the values as f['question'], f['answer'] and so on)
        # Also, do not forget to add a new line character (\n) at the end of each line.
        t += f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n" 
  

    return t


FAQ_LAYOUT = generate_faq_layout(FAQ)
print(FAQ_LAYOUT[:1000])


# GRADED CELL

def query_on_faq(query: str, **kwargs) -> dict:
    """
    Constructs a prompt to query an FAQ system and generates a response.

    Parameters:
    - query (str): The query about which the function seeks to provide an answer from the FAQ.
    - **kwargs: Optional keyword arguments for extra configuration of prompt parameters.

    Returns:
    - str: The response generated from the LLM based on the input query and FAQ layout.

    """
    ### START CODE HERE ###

    # Make the prompt. Don't forget to add the FAQ_LAYOUT and the query in it!
    prompt = (
        "You are a helpful customer support assistant. "
        "Based on the following FAQ information, answer the user's question.\n\n"
        f"FAQ:\n{FAQ_LAYOUT}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )

    # Generate the parameters dict with PROMPT and **kwargs 
    kwargs = generate_params_dict(prompt, **kwargs)

    ### END CODE HERE ###
    
    return kwargs

kwargs = query_on_faq("I got my cloth but I didn't like it. How can I return it?")

content = generate_with_single_input(**kwargs)
print(content['content'])
unittests.test_query_on_faq(query_on_faq)

# GRADED CELL

def decide_task_nature(query: str) -> str:
    """
    Determines whether a query is creative or technical.

    This function constructs a prompt for an LLM to decide if a given query requires a creative response,
    such as making suggestions or composing ideas, or a technical response, such as providing product details or prices.

    Parameters:
    - query (str): The query to be evaluated for its nature.

    Returns:
    - str: The label 'creative' if the query requires creative input, or 'technical' if it requires technical information.
    """

    ### START CODE HERE ###

    # Create the prompt. Remember to include the query, examples, and clear instructions (not necessarily in this order!)
    prompt = (
        "Classify the following query as either 'creative' or 'technical'. "
        "A creative query requires suggestions, composing outfits, or generating ideas. "
        "A technical query requires factual product details, prices, or specific catalog information.\n"
        "Return ONLY the word 'creative' or 'technical'.\n\n"
        "Examples:\n"
        "Query: Create a look suitable for a beach party. Label: creative\n"
        "Query: What are the most expensive items you have? Label: technical\n"
        "Query: Suggest an outfit for a rainy day. Label: creative\n"
        "Query: Give me three blue T-shirts from your catalogue. Label: technical\n\n"
        f"Query: {query} Label:"
    )

    # Generate the kwargs dictionary by passing the PROMPT, setting temperature to 0 and max_tokens to 1
    kwargs = generate_params_dict(prompt, temperature=0, max_tokens=1)

    # Generate the response using generate_with_single_input and **kwargs
    response = generate_with_single_input(**kwargs)

    # Get the label
    label = response['content'].strip().lower()

    ### END CODE HERE ###
    
    return label

queries = ["Give me two sneakers with vibrant colors.",
           "What are the most expensive clothes you have in your catalogue?",
           "I have a green dress and I like a suggestion on an accessory to match with it.",
           "Give me three trousers with vibrant colors you have in your catalogue.",
           "Create a look for a woman walking in a park on a sunny day. It must be fresh due to hot weather."
           ]

for query in queries:
    label = decide_task_nature(query)
    print(f"Query: {query} Label: {label}")

unittests.test_decide_task_nature(decide_task_nature)

# GRADED CELL 

def get_params_for_task(task: str) -> dict:
    """
    Retrieves specific LLM parameters based on the nature of the task.

    This function returns parameter sets optimized for either creative or technical tasks.
    Creative tasks benefit from higher randomness, while technical tasks require more focus and precision.
    A default parameter set is returned for unrecognized task types.

    Parameters:
    - task (str): The nature of the task ('creative' or 'technical').

    Returns:
    - dict: A dictionary containing 'top_p' and 'temperature' settings appropriate for the task.
    """
    ### START CODE HERE ###
    # Define the parameter sets for technical and creative tasks
    PARAMETERS_DICT = {
        "creative": {"top_p": 0.9, 'temperature': 1.2},
        "technical": {'top_p': 0.3, 'temperature': 0.2}
    }
    
    # Return the corresponding parameter set based on task type
    if task == 'technical':
        param_dict = PARAMETERS_DICT["technical"]
    elif task == 'creative':
        param_dict = PARAMETERS_DICT["creative"]
    else:
        # Fallback to a default parameter set for unrecognized task types
        param_dict = {"top_p": 0.5, "temperature": 0.5}
    ### END CODE HERE ###
    
    return param_dict

get_params_for_task("technical")

unittests.test_get_params_for_task(get_params_for_task)

# Let's remember the data structure of a product
print(PRODUCTS_DATA[0])
# Run this cell to generate the dictionary with the possible values for each key
values = {}
for d in PRODUCTS_DATA:
    for key, val in d.items():
        if key in ('product_id', 'price', 'productDisplayName', 'subCategory', 'year'):
            continue
        if key not in values.keys():
            values[key] = set()
        values[key].add(val)

# Example of possible values for the feature 'season'
print(values['season'])

# GRADED CELL

def generate_metadata_from_query(query: str) -> str:
    """
    Generates metadata in JSON format based on a given query to filter clothing items.

    This function constructs a prompt for an LLM to produce a JSON object
    that will guide filtering in a vector database query for clothing items.
    It uses possible values from a predefined set and ensures that only relevant metadata
    is included in the output JSON.

    Parameters:
    - query (str): A description of specific clothing-related needs.

    Returns:
    - str: A JSON string representing metadata with keys such as gender, masterCategory,
      articleType, baseColour, price, usage, and season. Each value in the JSON is a list.
      The price is specified as a dictionary with "min" and "max" keys.
      For unrestricted categories, use ["Any"], and if no price is specified,
      default to {"min": 0, "max": "inf"}.
    """
    ### START CODE HERE ### 

    # Construct the prompt.
    # Include the query, the desired JSON format, and the possible values (pass {values} where needed).
    # Clearly instruct the LLM to include gender, masterCategory, articleType, baseColour, price, usage, and season as keys.
    # Specify that the price key must be a JSON object with "min" and "max" values (0 if no lower bound, "inf" if no upper bound).
    # If no price is set, default to min = None
    prompt = (
        "Generate a JSON object to filter clothing items based on the following query. "
        "The JSON must contain these keys: gender, masterCategory, articleType, baseColour, price, usage, season.\n"
        "Each value should be a list of applicable values. "
        "The price key must be a JSON object with 'min' and 'max' keys (use 0 if no lower bound, 'inf' if no upper bound).\n"
        "If a category is unrestricted, use [\"Any\"].\n"
        "If no price is specified, use {\"min\": 0, \"max\": \"inf\"}.\n\n"
        f"Possible values for each key: {values}\n\n"
        f"Query: {query}\n\n"
        "Return ONLY the JSON object, no additional text."
    )

    # Generate the response with generate_with_single_input using PROMPT, temperature=0 (low randomness), and max_tokens=1500
    response = generate_with_single_input(prompt, temperature=0, max_tokens=1500)

    # Extract the content from the response
    content = response['content']

    ### END CODE HERE ###
    
    return content

print(generate_metadata_from_query("Create a look for a man that suits a sunny day in the park. I don't want to spend more than 300 dollars on each piece."))

def parse_json_output(llm_output: str) -> dict:
    """
    Parses a string output from an LLM into a JSON object.

    This function attempts to clean and parse a JSON-formatted string produced by an LLM.
    The input string might contain minor formatting issues, such as unnecessary newlines or single quotes
    instead of double quotes. The function attempts to correct such issues before parsing.

    Parameters:
    - llm_output (str): The string output from the LLM that is expected to be in JSON format.

    Returns:
    - dict or None: A dictionary if parsing is successful, or None if the input string cannot be parsed into valid JSON.

    Exception Handling:
    - In case of a JSONDecodeError during parsing, an error message is printed, and the function returns None.
    """
    try:
        # Since the input might be improperly formatted, ensure any single quotes are removed
        llm_output = llm_output.replace("\n", '').replace("'",'').replace("}}", "}").replace("{{", "{")  # Remove any erroneous structures
        
        # Attempt to parse JSON directly provided it is a properly-structured JSON string
        parsed_json = json.loads(llm_output)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None

json_string = generate_metadata_from_query("Give me three blue dresses suitable for a wedding party, less than 200 dollars and at least 50 dollars")
json_output = parse_json_output(json_string)

print(json_output)

products_collection = client.collections.get('products')
len(products_collection)

def get_filter_by_metadata(json_output: dict | None = None):
    """
    Generate a list of Weaviate filters based on a provided metadata dictionary.

    Parameters:
    - json_output (dict) or None: Dictionary containing metadata keys and their values.

    Returns:
    - list[Filter] or None: A list of Weaviate filters, or None if input is None.
    """
    # If the input dictionary is None, return None immediately
    if json_output is None:
        return None

    # Define a tuple of valid keys that are allowed for filtering
    valid_keys = (
        'gender',
        'masterCategory',
        'articleType',
        'baseColour',
        'price',
        'usage',
        'season',
    )

    # Initialize an empty list to store the filters
    filters = []

    # Iterate over each key-value pair in the input dictionary
    for key, value in json_output.items():
        # Skip the key if it is not in the list of valid keys
        if key not in valid_keys:
            continue

        # Special handling for the 'price' key
        if key == 'price':
            # Ensure the value associated with 'price' is a dictionary
            if not isinstance(value, dict):
                continue

            # Extract the minimum and maximum prices from the dictionary
            min_price = value.get('min')
            max_price = value.get('max')

            # Skip if either min_price or max_price is not provided
            if min_price is None or max_price is None:
                continue

            # Skip if min_price is non-positive or max_price is infinity
            if min_price <= 0 or max_price == 'inf':
                continue

            # Add filters for price greater than min_price and less than max_price
            filters.append(Filter.by_property(key).greater_than(min_price))
            filters.append(Filter.by_property(key).less_than(max_price))
        else:
            # For other valid keys, add a filter that checks for any of the provided values
            filters.append(Filter.by_property(key).contains_any(value))

    return filters

def generate_filters_from_query(query: str) -> list:
    json_string = generate_metadata_from_query(query)
    json_output = parse_json_output(json_string)
    filters = get_filter_by_metadata(json_output)
    return filters

filters = generate_filters_from_query("Give me three T-shirts to use in sunny days")

print(filters)

def get_relevant_products_from_query(query: str):
    """
    Retrieve products that are most relevant to a given query by applying filters.

    This function generates filters based on the provided query and uses them to find 
    products that closely match the query criteria. If no filters are applicable or if 
    the initial search returns a small number of products, the function dynamically reduces 
    the filtering constraints based on a predefined order of filter importance.

    Parameters:
    query (str): The query string used to search for relevant products.

    Returns:
    list: A list of product objects that are most relevant to the query. If filters are not effective,
          it adjusts them to ensure a minimum return of products.
    """
    filters = generate_filters_from_query(query)  # Generate filters based on query

    # Check if there are no applicable filters
    if filters is None or len(filters) == 0:
        # Query the collection without filters, using the query text for relevance
        res = products_collection.query.near_text(query, limit=20).objects
        return res

    # Query with filters and limit to top 20 relevant objects
    res = products_collection.query.near_text(query, filters=Filter.all_of(filters), limit=20).objects

    # If the result set is fewer than 10 products, try reducing filters to broaden the search
    importance_order = ['baseColour', 'masterCategory', 'usage', 'masterCategory', 'season', 'gender']

    if len(res) < 10:
        # Iterate through the importance order of filters
        for i in range(len(importance_order)):
            # Create a list of filters that excludes less important ones
            filtered_filters = [x for x in filters if x.target not in importance_order[i+1:]]
            
            # Re-query with the reduced set of filters
            res = products_collection.query.near_text(query, filters=Filter.all_of(filtered_filters), limit=20).objects
            
            # If sufficient products have been found, return early
            if len(res) >= 5:
                return res
        # If there are no enough results, perform a basic near_text with only the query.
        if len(res) < 5:
            res = products_collection.query.near_text(query, limit=20).objects
        
    return res  # Return the final set of relevant products


query = "Give me three T-shirts to use in sunny days"
t = get_relevant_products_from_query("Give me three T-shirts to use in sunny days")
# Check if t is non-empty
if len(t) > 0:
    print(t[0].properties)

def generate_items_context(results: list) -> str:
    """
    Compile detailed product information from a list of result objects into a formatted string.

    This function takes a list of results, each containing various product attributes, and constructs 
    a human-readable summary for each product. Each product's details, including ID, name, category, 
    usage, gender, type, and other characteristics, are concatenated into a string that describes 
    all products in the list.

    Parameters:
    results (list): A list of result objects, each having a `properties` attribute that is a dictionary 
                    containing product attributes such as 'product_id', 'productDisplayName', 
                    'masterCategory', 'usage', 'gender', 'articleType', 'subCategory', 
                    'baseColour', 'season', and 'year'.

    Returns:
    str: A multi-line string where each line contains the formatted details of a single product.
         Each product detail includes the product ID, name, category, usage, gender, type, color, 
         season, and year.
    """
    t = ""  # Initialize an empty string to accumulate product information

    for item in results:  # Iterate through each item in the results list
        item = item.properties  # Access the properties dictionary of the current item

        # Append formatted product details to the output string
        t += (
            f"Product ID: {item['product_id']}. "
            f"Product name: {item['productDisplayName']}. "
            f"Product Category: {item['masterCategory']}. "
            f"Product usage: {item['usage']}. "
            f"Product gender: {item['gender']}. "
            f"Product Type: {item['articleType']}. "
            f"Product Category: {item['subCategory']} "
            f"Product Color: {item['baseColour']}. "
            f"Product Season: {item['season']}. "
            f"Product Year: {item['year']}.\n"
        )

    return t  # Return the complete formatted string with product details

print(generate_items_context(t)[:1000])

def query_on_products(query: str) -> dict:
    """
    Execute a product query process to generate a response based on the nature of the query.

    This function analyzes the type of query — whether it is technical or creative — and retrieves 
    relevant product information accordingly. It constructs a prompt that includes product details 
    and the original query, and then generates parameters for querying an LLM.
    Finally, it generates a response based on the prompt and returns the content of the response.

    Parameters:
    query (str): The input query string that needs to be analyzed and answered using product data.

    Returns:
    dict: A dictionary of keyword arguments (`kwargs`) containing the prompt and additional settings 
          for creating a response, suitable for input to an LLM or other processing system.

    Outputs:
    dict: A dictionary with the parameters to call an LLM
    """


    # Determine if the query is technical or creative in nature
    query_label = decide_task_nature(query) 
    
    # Obtain necessary parameters based on the query type
    parameters_dict = get_params_for_task(query_label) 
    
    # Retrieve products that are relevant to the query
    relevant_products = get_relevant_products_from_query(query) 
     
    # Create a context string from the relevant products
    context = generate_items_context(relevant_products) 

    # Construct a prompt including product details and the query. Remember to add the context and the query in the prompt, also, ask the LLM to provide the product ID in the answer
    prompt = (
    f"Given the available set of cloth products, answer the question that follows, providing the item ID in your answers. "
    f"Other information might be provided but not necessarily all of them; pick only the relevant ones for the given query and avoid being too long when describing the items' features. "
    f"If no number of products is mentioned in the query, select at most five to show. "
    f"CLOTH PRODUCTS AVAILABLE: {context} "
    f"QUERY: {query}"
        )
    
    # Generate kwargs (parameters dict) for parameterized input to the LLM with , Prompt, role = 'assistant' and **parameters_dict
    kwargs = generate_params_dict(prompt, role='assistant', **parameters_dict)
    
    
    return kwargs

kwargs = query_on_products('Make a wonderful look for a man attending a wedding party happening during night.')
result = generate_with_single_input(**kwargs)
print(result['content'])

kwargs = query_on_products('Give me three T-shirts for sunny days')
result = generate_with_single_input(**kwargs)
print(result['content'])

def answer_query(query: str) -> dict:
    """
    Determines the type of a given query (FAQ or Product) and executes the appropriate workflow.

    Parameters:
    - query (str): The user's query string.

    Returns:
    - dict: A dictionary of keyword arguments to be used for further processing.
      If the query is neither FAQ nor Product-related, returns a default response dictionary
      instructing the assistant to answer based on existing context.
    """
    label = check_if_faq_or_product(query)
    if label not in ['FAQ', 'Product']:
        return {
            "role": "assistant",
            "prompt": f"User provided a question that does not fit FAQ or Product related questions. "
                      f"Answer it based on the context you already have so far. Query provided by the user: {query}"
        }
    if label == 'FAQ':
        kwargs = query_on_faq(query)
    if label == 'Product':
        try:
            kwargs = query_on_products(query)
        except:
            return {
            "role": "assistant",
            "prompt": f"User provided a question that broke the querying system. Instruct them to rephrase it."
                      f"Answer it based on the context you already have so far. Query provided by the user: {query}"
        }
            
    return kwargs

kwargs = answer_query("What are your working hours?")

result = generate_with_single_input(**kwargs)
print(result['content'])

chat_widget = ChatWidget(generator_function = answer_query)