import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_chat_completion, get_completion_from_messages
from unitls import read_string_to_list, generate_output_string

dash_line = '-' *  100
delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Output a python list of objects, where each object has \
the following format:
    'category': <one of Computers and Laptops, \
    Smartphones and Accessories, \
    Televisions and Home Theater Systems, \
    Gaming Consoles and Accessories, 
    Audio Equipment, Cameras and Camcorders>,
OR
    'products': <a list of products that must \
    be found in the allowed products below>

Where the categories and products must be found in \
the customer service query.
If a product is mentioned, it must be associated with \
the correct category in the allowed products list below.
If no products or categories are found, output an \
empty list.

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

Only output the list of objects, with nothing else.
"""
user_message_1 = f"""
 tell me about the smartx pro phone and \
 the fotosnap camera, the dslr one. \
 Also tell me about your tvs """
messages =  [  
{'role':'system', 
 'content': system_message},    
{'role':'user', 
 'content': f"{delimiter}{user_message_1}{delimiter}"},  
] 
category_and_product_response_1 = get_completion_from_messages(messages)

print("### Extract relevant product and category names")
print(dash_line)
print(user_message_1)
print(dash_line)
print(category_and_product_response_1)

print("\n\n")
print(dash_line)
print("Read Python string into Python list of dictionaries")
category_and_product_list = read_string_to_list(category_and_product_response_1)
print(category_and_product_list)

print("\n\n")
print(dash_line)
print("Retrieve detailed product information for the relevant products and categories")
print(dash_line)

product_info_for_user_message_1 = generate_output_string(category_and_product_list)
print(product_info_for_user_message_1)

print("\n\n")
print(dash_line)
print("Generate answer to user query based on detailed product information")
print(dash_line)

system_message_2 = f"""
You are a helpful customer service assistant for a \
large electronics store.
Respond in a friendly and helpful tone \
with very concise answers. \
Make sure to ask the user relevant follow-up questions. \
"""

user_message_2 = f"""
tell me about the smartx pro phone and \
the fotosnap camera, the dslr one. \
Also tell me about your tvs
"""
messages_2 =  [
    {"role": "system", "content": system_message_2},
    {"role": "user", "content": f"{delimiter}{user_message_2}{delimiter}"},
    {"role": "assistant", "content": f"{delimiter}{product_info_for_user_message_1}{delimiter}"}
]

final_response = get_completion_from_messages(messages_2)
print(final_response)