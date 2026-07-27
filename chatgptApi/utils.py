import json
from pathlib import Path
from collections import defaultdict
from aiSetup import get_completion_from_messages

products_file = 'products.json'
categories_file = 'categories.json'
MODULE_DIR = Path(__file__).resolve().parent
PRODUCTS_PATH = MODULE_DIR / "products.json"
CATEGORIES_PATH = MODULE_DIR / "categories.json"

def load_json_file(file_path, default=None):
    """Load JSON from disk and return default when the file is missing or invalid."""
    path = Path(file_path)
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def read_string_to_list(input_string):
    """Convert a string representation of a list into an actual list."""
    if input_string is None:
        return None

    try:
        normalized = input_string.replace("'", '"')
        data = json.loads(normalized)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
        return None

def _load_products_payload():
    return load_json_file(PRODUCTS_PATH, default={})


def _load_categories_payload():
    return load_json_file(
        CATEGORIES_PATH,
        default={"categories": [], "category_to_products": {}},
    )


def get_products_by_name(name):
    products = _load_products_payload()
    return products.get(name, None)


def get_products_by_category(category):
    products = _load_products_payload()
    return [product for product in products.values() if product.get("category") == category]


def get_categories():
    categories_payload = _load_categories_payload()
    return categories_payload.get("categories", [])


def get_category_to_products():
    categories_payload = _load_categories_payload()
    return categories_payload.get("category_to_products", {})


def generate_output_string(data_list):
    output_string = ""

    if data_list is None:
        return output_string

    for data in data_list:
        try:
            if "products" in data:
                products_list = data["products"]
                for product_name in products_list:
                    product = get_products_by_name(product_name)
                    if product:
                        output_string += json.dumps(product, indent=4) + "\n"
                    else:
                        print(f"Error: Product '{product_name}' not found")
            elif "category" in data:
                category_name = data["category"]
                category_products = get_products_by_category(category_name)
                for product in category_products:
                    output_string += json.dumps(product, indent=4) + "\n"
            else:
                print("Error: Invalid object format")
        except Exception as e:
            print(f"Error: {e}")

    return output_string

def get_categories():
    with open(categories_file, 'r') as file:
            categories = json.load(file)
    return categories


def get_product_list():
    """
    Used in L4 to get a flat list of products
    """
    products = get_products()
    product_list = []
    for product in products.keys():
        product_list.append(product)
    
    return product_list

def get_products_and_category():
    """
    Used in L5
    """
    products = get_products()
    products_by_category = defaultdict(list)
    for product_name, product_info in products.items():
        category = product_info.get('category')
        if category:
            products_by_category[category].append(product_info.get('name'))
    
    return dict(products_by_category)

def get_products():
    with open(products_file, 'r') as file:
        products = json.load(file)
    return products

def find_category_and_product(user_input,products_and_category):
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    OR
        'products': <a list of products that must be found in the allowed products below>

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}
    
    """
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return get_completion_from_messages(messages)

def find_category_and_product_only(user_input,products_and_category):
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of objects, where each object has the following format:
    'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    OR
    'products': <a list of products that must be found in the allowed products below>

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    Allowed products: 
    Computers and Laptops category:
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Smartphones and Accessories category:
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Televisions and Home Theater Systems category:
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Gaming Consoles and Accessories category:
GameSphere X
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Audio Equipment category:
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Cameras and Camcorders category:
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera
    
    Only output the list of objects, nothing else.
    """
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_input}{delimiter}"},  
    ] 
    return get_completion_from_messages(messages)

def get_products_from_query(user_msg):
    """
    Code from L5, used in L8
    """
    products_and_category = get_products_and_category()
    delimiter = "####"
    system_message = f"""
    You will be provided with customer service queries. \
    The customer service query will be delimited with {delimiter} characters.
    Output a python list of json objects, where each object has the following format:
        'category': <one of Computers and Laptops, Smartphones and Accessories, Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, Audio Equipment, Cameras and Camcorders>,
    OR
        'products': <a list of products that must be found in the allowed products below>

    Where the categories and products must be found in the customer service query.
    If a product is mentioned, it must be associated with the correct category in the allowed products list below.
    If no products or categories are found, output an empty list.

    The allowed products are provided in JSON format.
    The keys of each item represent the category.
    The values of each item is a list of products that are within that category.
    Allowed products: {products_and_category}

    """
    
    messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_msg}{delimiter}"},  
    ] 
    category_and_product_response = get_completion_from_messages(messages)
    
    return category_and_product_response


def get_mentioned_product_info(category_and_product_list):
    """
    Takes a list of products/categories and returns detailed product information.
    Uses the generate_output_string function to format the product details.
    """
    return generate_output_string(category_and_product_list)


def answer_user_msg(user_msg, product_info):
    """
    Generate a customer service response to the user's message using product information.
    
    Args:
        user_msg (str): The user's question or request
        product_info (str): Formatted product information to use in the response
    
    Returns:
        str: The assistant's response to the user
    """
    system_message = """
You are a customer service assistant for a large electronic store. \
Respond in a friendly and helpful tone, with concise answers. \
Make sure to ask the user relevant follow-up questions.
"""
    
    user_message = f"""
Here is information about products the customer is asking about:
{product_info}

Please provide a helpful response to the customer's question:
{user_msg}
"""
    
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    
    response = get_completion_from_messages(messages)
    return response

