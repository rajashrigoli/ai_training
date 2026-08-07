import os
from dotenv import load_dotenv
load_dotenv("/Users/rajashri.samal/ai_training/.env")

from typing import List
import requests
import re
import weaviate
from weaviate.classes.config import Configure, Property, DataType, Tokenization
from weaviate.util import generate_uuid5
import tqdm
from weaviate.classes.query import Filter

from utils import (
    generate_with_single_input, 
    suppress_subprocess_output,
    kill_processes_on_ports
)

# Kill processes on ports before importing flask_app
# WARNING: Running this cell twice may kill the active kernel
kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

import flask_app

dash_line = "-" * 100

print(dash_line)
print("Downloading the data")
print(dash_line)

url = "https://raw.githubusercontent.com/progit/progit2/main/book/01-introduction/sections/what-is-git.asc"
source_text = requests.get(url).text

print(source_text[:1000])

print(f"There are about {len(source_text.split())} words in this chapter. Depending on how your LLM tokenizes words, you'd expect roughly {round(len(source_text.split())*1.3)} tokens.")

print("\n\n")
print(dash_line)
print("2 - Fixed-size chunking")
print(dash_line)

print("\n\n")
print("2.1 Example Chunking Code")
print(dash_line)

def get_chunks_fixed_size(text: str, chunk_size: int) -> List[str]:
    """
    Splits a given text into chunks of a specified fixed size.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of words per chunk.

    Returns:
        List[str]: A list of text chunks, each containing up to 'chunk_size' words.
    """
    # Split the input text into individual words
    text_words = text.split()
    
    # Initialize a list to hold the chunks of words
    chunks = []
    
    # Iterate over the word indices in steps of 'chunk_size'
    for i in range(0, len(text_words), chunk_size):
        # Select a sublist of words from 'i' to 'i + chunk_size'
        chunk_words = text_words[i: i + chunk_size]
        
        # Join the selected words into a single string with spaces in between
        chunk = " ".join(chunk_words)
        
        # Add the chunk to the list of chunks
        chunks.append(chunk)
    
    # Return the list of word chunks
    return chunks


fixed_size_chunks = get_chunks_fixed_size(source_text, chunk_size = 100)
print(len(fixed_size_chunks))
fixed_size_chunks[0:3]

print("\n\n")
print("2.2 Chunking with overlap")
print(dash_line)
def get_chunks_fixed_size_with_overlap(text: str, chunk_size: int, overlap_fraction: float) -> List[str]:
    """
    Splits a given text into chunks of a fixed size with a specified overlap fraction between consecutive chunks.

    Parameters:
    - text (str): The input text to be split into chunks.
    - chunk_size (int): The number of words each chunk should contain.
    - overlap_fraction (float): The fraction of the chunk size that should overlap with the adjacent chunk.
      For example, an overlap_fraction of 0.2 means 20% of the chunk size will be used as overlap.

    Returns:
    - List[str]: A list of chunks (each a string) where each chunk might overlap with its adjacent chunk.
    """

    # Split the text into individual words
    text_words = text.split()
    
    # Calculate the number of words to overlap between consecutive chunks
    overlap_int = int(chunk_size * overlap_fraction)
    
    # Initialize a list to store the resulting chunks
    chunks = []
    
    # Iterate over text in steps of chunk_size to create chunks
    for i in range(0, len(text_words), chunk_size):
        # Determine the start and end indices for the current chunk,
        # taking into account the overlap with the previous chunk
        chunk_words = text_words[max(i - overlap_int, 0): i + chunk_size]
        
        # Join the selected words to form a chunk string
        chunk = " ".join(chunk_words)
        
        # Append the chunk to the list of chunks
        chunks.append(chunk)
    
    # Return the list of chunks
    return chunks

for chosen_size in [5, 25, 100]:
    chunks = get_chunks_fixed_size_with_overlap(source_text, chosen_size, overlap_fraction=0.2)
    # Print outputs to screen
    print(f"\nSize {chosen_size} - {len(chunks)} chunks returned.")
    for i in range(3):
        print(f"Chunk {i+1}: {chunks[i]}")


print("\n\n")
print(dash_line)
print("3 - Variable-size chunking - Recursive Character Splitting")
print(dash_line)

print("\n\n")
print("3.1 Pseudo-code for variable-size chunking methods")
print(dash_line)

# Split the text into paragraphs
def get_chunks_by_paragraph(source_text: str) -> List[str]:
    return source_text.split("\n\n")

# Split the text by Asciidoc section markers
def get_chunks_by_asciidoc_sections(source_text: str) -> List[str]:
    return source_text.split("\n==")

for marker in ["\n\n", "\n=="]:
    chunks = source_text.split(marker)
    # Print outputs to screen
    print(f"\nUsing the marker: {repr(marker)} - {len(chunks)} chunks returned.")
    for i in range(3):
        print(f"Chunk {i+1}: {repr(chunks[i])}")



print("\n\n")
print("3.2 Mixing fixed and variable-sized chunking")
print(dash_line)

def mixed_chunking(source_text):
    """
    Splits the given source_text into chunks using a mix of fixed-size and variable-size chunking.
    It first splits the text by Asciidoc markers and then processes the chunks to ensure they are 
    of appropriate size. Smaller chunks are merged with the next chunk, and larger chunks can be 
    further split at the middle or specific markers within the chunk.

    Args:
    - source_text (str): The text to be chunked.

    Returns:
    - list: A list of text chunks.
    """

    # Split the text by Asciidoc marker
    chunks = source_text.split("\n==")

    # Chunking logic
    new_chunks = []
    chunk_buffer = ""
    min_length = 25

    for chunk in chunks:
        new_buffer = chunk_buffer + chunk  # Create new buffer
        new_buffer_words = new_buffer.split(" ")  # Split into words
        if len(new_buffer_words) < min_length:  # Check whether buffer length is too small
            chunk_buffer = new_buffer  # Carry over to the next chunk
        else:
            new_chunks.append(new_buffer)  # Add to chunks
            chunk_buffer = ""

    if len(chunk_buffer) > 0:
        new_chunks.append(chunk_buffer)  # Add last chunk, if necessary

    return new_chunks


mixed_chunks = mixed_chunking(source_text)
for i in range(3):
    print(f"Chunk {i+1}: {repr(mixed_chunks[i])}")

print("\n\n")
print(dash_line)
print("4 - Chunking on real data")
print(dash_line)

print("\n\n")
print("4.1 Getting the data")
print(dash_line)

def get_book_text_objects():
    # Source location
    text_objs = list()
    api_base_url = 'https://api.github.com/repos/progit/progit2/contents/book'  # Book base URL
    chapter_urls = ['/01-introduction/sections', '/02-git-basics/sections']  # List of section URLs

    # Use a session without .netrc auth to avoid bad credentials
    session = requests.Session()
    session.trust_env = False  # Ignore .netrc credentials

    # Loop through book chapters
    for chapter_url in chapter_urls:
        response = session.get(api_base_url + chapter_url)  # Get the JSON data for the section files in the chapter

        # Check if response is valid
        if response.status_code != 200:
            print(f"Warning: Failed to fetch {chapter_url} (status {response.status_code}): {response.text[:200]}")
            continue

        data = response.json()
        if not isinstance(data, list):
            print(f"Warning: Unexpected response for {chapter_url}: {str(data)[:200]}")
            continue

        # Loop through inner files (sections)
        for file_info in data:
            if file_info['type'] == 'file':  # Only process files (not directories)
                file_response = session.get(file_info['download_url'])

                # Build objects including metadata
                chapter_title = file_info['download_url'].split('/')[-3]
                filename = file_info['download_url'].split('/')[-1]
                text_obj = {
                    "body": file_response.text,
                    "chapter_title": chapter_title,
                    "filename": filename
                }
                text_objs.append(text_obj)
    return text_objs


# This will generate a list with 14 elements, one for each chapter
book_text_objs = get_book_text_objects()

print(book_text_objs[0].keys())

print("\n\n")
print("4.2 Chunking the chapters")
print(dash_line)

def build_chunk_objs(book_text_obj, chunks):
    """
    Constructs a list of chunk objects from a given book text object 
    and its associated chunks.

    Args:
        book_text_obj (dict): A dictionary containing metadata for the book text, 
                              including 'chapter_title' and 'filename'.
        chunks (list): A list of chunks that represent parts of the book text.

    Returns:
        list: A list of dictionaries, each representing a chunk object 
              with 'chapter_title', 'filename', 'chunk', and 'chunk_index'.
    """
    chunk_objs = list()  # Initialize an empty list to store chunk objects
    
    # Iterate over the chunks with an index
    for i, c in enumerate(chunks):
        # Create a dictionary for each chunk with its associated data
        chunk_obj = {
            "chapter_title": book_text_obj["chapter_title"],  # Chapter title from the book text object
            "filename": book_text_obj["filename"],            # Filename from the book text object
            "chunk": c,                                       # The actual chunk of text
            "chunk_index": i                                  # The index of the chunk in the list
        }
        # Append the constructed chunk object to the list
        chunk_objs.append(chunk_obj)

    # Return the list of chunk objects
    return chunk_objs

# Get multiple sets of chunks - according to chunking strategy
chunk_obj_sets = dict()
for book_text_obj in book_text_objs:
    text = book_text_obj["body"]  # Get the object's text body

    # Loop through chunking strategies:
    for strategy_name, chunks in [
        ["fixed_size_25", get_chunks_fixed_size_with_overlap(text, 25, 0.2)],
        ["fixed_size_100", get_chunks_fixed_size_with_overlap(text, 100, 0.2)],
        ["para_chunks", get_chunks_by_paragraph(text)],
        ["para_chunks_min_25", mixed_chunking(text)]
    ]:
        chunk_objs = build_chunk_objs(book_text_obj, chunks)

        if strategy_name not in chunk_obj_sets.keys():
            chunk_obj_sets[strategy_name] = list()

        chunk_obj_sets[strategy_name] += chunk_objs

print(chunk_obj_sets.keys())
chunk_type = 'fixed_size_25' # Change it to check the different chunks!
print(chunk_obj_sets[chunk_type][0:2])

print("\n\n")
print("4.3 Loading Chunks into a Vector Database")
print(dash_line)

# Killing processes that might be listening in ports where Weaviate runs
kill_processes_on_ports([8080, 8079, 50050, 50051])

# Loading the client
collection_base = os.getenv('COLLECTION_M3', './')
persistence_path = os.path.join(collection_base, 'ungraded_lab_2')

with suppress_subprocess_output():
    try:
        client = weaviate.connect_to_embedded(
            persistence_data_path=persistence_path,
            environment_variables={
                "ENABLE_API_BASED_MODULES": "true", # Enable API based modules
                "ENABLE_MODULES": 'text2vec-transformers', # We will be using a transformer model
                "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5000/", # The endpoint the weaviate API will be using to vectorize
            }
        )
    except Exception as e:
        print(f"Failed to connect to embedded Weaviate: {e}")
        print("Falling back to local Weaviate connection...")
        try:
            client = weaviate.connect_to_local(port=8079, grpc_port=50050)
        except Exception as e2:
            print(f"Failed to connect to local Weaviate: {e2}")
            print("Trying alternative ports...")
            client = weaviate.connect_to_local(port=8080, grpc_port=50051)


# Creating the collection
if not client.collections.exists("chunking_example"):
    collection = client.collections.create(
            name='chunking_example',

            vectorizer_config=[Configure.NamedVectors.text2vec_transformers(
                    name="vector", # This is the name you will need to access the vectors of the objects in your collection
                    #source_properties=['chunk'], # which properties should be used to generate a vector, they will be appended to each other when vectorizing
                    vectorize_collection_name = False, # This tells the client to not vectorize the collection name. 
                                                       # If True, it will be appended at the beginning of the text to be vectorized
                    inference_url="http://127.0.0.1:5000", # Since we are using an API based vectorizer, you need to pass the URL used to make the calls 
                                                           # This was setup in our Flask application
                )],

            properties=[  # Define properties
            Property(name="chunk",data_type= DataType.TEXT),
            Property(name="chapter_title", data_type=DataType.TEXT),
            Property(name="filename",data_type=DataType.TEXT),
            Property(name="chunking_strategy",data_type=DataType.TEXT, tokenization = Tokenization.FIELD), # tokenization = Tokenization.FIELD means that the entire word will be treated as a token,
            Property(name="chunk_index",data_type=DataType.INT),

        ]
        )
else:
    collection = client.collections.get("chunking_example")

print(f"Total count: {collection.aggregate.over_all().total_count}")
for chunking_strategy in chunk_obj_sets.keys():
    where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy) # Filter by chunking strategy
    count = collection.aggregate.over_all(filters = where_filter).total_count # Aggregate with filtering
    print(f"Object count for {chunking_strategy}: {count}")


print("\n\n")
print(dash_line)
print("5 - Searching")
print(dash_line)

search_string = "history of git"  # Or "available git remote commands"

for chunking_strategy in chunk_obj_sets.keys():
    where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy)
    response = collection.query.near_text(search_string, filters = where_filter, limit = 2)
    print(f"RETRIEVED OBJECTS FOR CHUNKING STRATEGY {chunking_strategy.upper()}:\n")
    for i, obj in enumerate(response.objects):
        print(f"===== Object {i} =====")
        print(f"{obj.properties['chunk']}")
        print()

search_string = "how to add the url of a remote repository"  # Or "available git remote commands"

for chunking_strategy in chunk_obj_sets.keys():
    where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy)
    response = collection.query.near_text(search_string, filters = where_filter, limit = 2)
    print(f"RETRIEVED OBJECTS FOR CHUNKING STRATEGY {chunking_strategy.upper()}:\n")
    for i, obj in enumerate(response.objects):
        print(f"===== Object {i} =====")
        print(f"{obj.properties['chunk']}")
        print()


print("\n\n")
print(dash_line)
print("6 - Incorporating in a RAG system")
print(dash_line)

PROMPT = "Using this information and only this information, please explain {search_string} in a few short points.\nContext: {context}"

# Set number of chunks to retrieve to compensate for different chunk sizes

n_chunks_by_strat = dict()

# Grab more of shorter chunks
n_chunks_by_strat['fixed_size_25'] = 8
n_chunks_by_strat['para_chunks'] = 8

# Grab fewer of longer chunks
n_chunks_by_strat['fixed_size_100'] = 2
n_chunks_by_strat['para_chunks_min_25'] = 2

# Perform Retreval augmented generation
search_string = "history of git"  # Or "available git remote commands"

for chunking_strategy in chunk_obj_sets.keys():
    where_filter = Filter.by_property('chunking_strategy').equal(chunking_strategy)
    response = collection.query.near_text(search_string, filters = where_filter, limit = n_chunks_by_strat[chunking_strategy])
    context_string = ""
    for obj in response.objects:
        context_string += obj.properties['chunk'] + '\n'
    prompt = PROMPT.format(search_string = search_string, context = context_string)
    response = generate_with_single_input(prompt, role = 'assistant')
    print(f"Search string: {search_string}")
    print(f"Chunking Strategy: {chunking_strategy}:")
    print(f"Response:\n\t{response['content']}")
    print()

    # Don't forget to close the client!
client.close()