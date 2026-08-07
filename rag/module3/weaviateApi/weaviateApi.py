from dotenv import load_dotenv
load_dotenv("/Users/rajashri.samal/ai_training/.env")

from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from typing import List
from tqdm import tqdm
import joblib
import weaviate
import re
from weaviate.util import generate_uuid5
from pprint import pprint
import os

from weaviate.classes.query import Rerank

from utils import (
    suppress_subprocess_output, 
    generate_with_single_input, 
    print_object_properties,
    kill_processes_on_ports
)

# Kill processes on ports BEFORE importing flask_app (which starts Flask on port 5000)
# WARNING: Running this cell twice may kill the active kernel
kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

import flask_app

dash_line = "-" * 100

print(dash_line)
print("The Weaviate Client")
print(dash_line)
client = weaviate.connect_to_embedded(
    version="1.27.6",
    persistence_data_path="./.collections",
    headers={
        "X-OpenAI-Api-Key": os.environ["GROQ_API_KEY"]
    },
    environment_variables={
        "ENABLE_API_BASED_MODULES": "true",
        "ENABLE_MODULES": 'text2vec-transformers, reranker-transformers',
        "TRANSFORMERS_INFERENCE_API":"http://127.0.0.1:5000/",
        "RERANKER_INFERENCE_API":"http://127.0.0.1:5000/"
    }
)

print("\n\n")
print(dash_line)
print("Configuring the database:")
print(dash_line)

print("\n\n")
print("2.1 Creating a Collection")
data = joblib.load("data.joblib")
print_object_properties(data[0])

print("\n\n")
print("2.2 Configuring the Vectorizer")
print(dash_line)

vectorizer_config = [Configure.NamedVectors.text2vec_transformers(
                name="vector", # This is the name you will need to access the vectors of the objects in your collection
                source_properties=['place', 'state', 'description', 'best_season_to_visit', 'attractions', 'budget'], # which properties should be used to generate a vector, they will be appended to each other when vectorizing
                vectorize_collection_name = False, # This tells the client to not vectorize the collection name. 
                                                   # If True, it will be appended at the beginning of the text to be vectorized
                inference_url="http://127.0.0.1:5000", # Since we are using an API based vectorizer, you need to pass the URL used to make the calls 
                                                       # This was setup in our Flask application
            )]



print("\n\n")
print("2.3 The Properties")
print(dash_line)

# Delete the collection in case it exists
if client.collections.exists("example_collection"):
    client.collections.delete("example_collection")
    

if not client.collections.exists('example_collection'): # Creates only if the collection does not exist
    collection = client.collections.create(
            name='example_collection',
            vectorizer_config=vectorizer_config, # The config we defined before,
            reranker_config=Configure.Reranker.transformers(), # The reranker config

            properties=[  # Define properties
            Property(name="place",vectorize_property_name=True,data_type= DataType.TEXT),
            Property(name="state",vectorize_property_name=True, data_type=DataType.TEXT),
            Property(name="description",vectorize_property_name=True, data_type=DataType.TEXT),
            Property(name="best_season_to_visit",vectorize_property_name=True, data_type=DataType.TEXT),
            Property(name="attractions",vectorize_property_name=True, data_type=DataType.TEXT),
            Property(name="budget",vectorize_property_name=True, data_type=DataType.TEXT),
            Property(name="user_ratings", data_type=DataType.NUMBER),
            Property(name="last_updated", data_type=DataType.DATE),

        ]
        )
else:
    collection = client.collections.get("example_collection")


print(collection)

try:
    collection = client.collections.create(
        name='example_collection',

        vectorizer_config=vectorizer_config, # The config we defined before,
    
        properties=[  # Define properties
        Property(name="place",vectorize_property_name=True,data_type= DataType.TEXT),
        Property(name="state",vectorize_property_name=True, data_type=DataType.TEXT),
        Property(name="description",vectorize_property_name=True, data_type=DataType.TEXT),
        Property(name="best_season_to_visit",vectorize_property_name=True, data_type=DataType.TEXT),
        Property(name="attractions",vectorize_property_name=True, data_type=DataType.TEXT),
        Property(name="budget",vectorize_property_name=True, data_type=DataType.TEXT),
        Property(name="user_ratings", data_type=DataType.NUMBER),
        Property(name="last_updated", data_type=DataType.DATE),
                 
    ]
    )
except Exception as e:
    print(e)

client.collections.list_all().keys()


print("\n\n")
print("2.4 Adding elements into a Collection")
print(dash_line)

# Set up a batch process with specified fixed size and concurrency
with collection.batch.fixed_size(batch_size=1, concurrent_requests=1) as batch:
    # Iterate over a subset of the dataset
    for document in tqdm(data): # tqdm is a library to show progress bars
            # Generate a UUID based on the article_content text for unique identification
            uuid = generate_uuid5(document)

            # Add the object to the batch with properties and UUID. 
            # properties expects a dictionary with the keys being the properties.
            batch.add_object(
                properties=document,
                uuid=uuid,
            )

print(len(collection))

print("\n\n")
print(dash_line)
print("3 - Querying on a collection")
print(dash_line)

print("\n\n")
print("3.1 Filters")
print(dash_line)
# Here we are fetching 2 objects with a filter by property, filtering by 'user_ratings, only objects with value greater or equal to 3.5'
result = collection.query.fetch_objects(limit = 2, filters = Filter.by_property('user_ratings').greater_or_equal(3.5))

print(result)
print(result.objects)

print("\n\n")
print("3.2 Semantic Search")
print(dash_line)

result = collection.query.near_text(query = 'I want suggestions to travel during Winter. I want cheap places.', limit = 4)
# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)

result = collection.query.near_text(query = 'I want suggestions to travel during Winter. I want cheap places.', 
                                    filters = Filter.by_property('budget').equal('Low'),
                                    limit = 4)

# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)


result = collection.query.near_text(query = 'I want suggestions to travel during Winter. I want cheap places.', 
                                    filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
                                    limit = 4)



# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)

print("\n\n")
print("3.3 BM25 search")
print(dash_line)

result = collection.query.bm25(query = 'I want suggestions to travel during Winter. I want cheap places.', 
                                    filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
                                    limit = 4)


# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)

print("\n\n")
print("3.4 Hybrid Search")
print(dash_line)

result = collection.query.hybrid(query = 'I want suggestions to travel during Winter. I want cheap places.', 
                                    filters = Filter.by_property('budget').contains_any(['Low','Moderate']),
                                    alpha = 0.3,
                                    limit = 4)


# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)


print("\n\n")
print("3.5 Reranking Search")
print(dash_line)
response = collection.query.near_text(
    query="'I want suggestions to travel during Winter. I want cheap and fun places.'",  
    limit=5,
    rerank=Rerank(
        prop="attractions",                   # The property to rerank on
        query="Fun places"  # If not provided, the original query will be used
    )
)

# Let's iterate over the result objects and return their properties
for obj in result.objects:
    print_object_properties(obj.properties)

# Don't forget to close the client!
client.close()