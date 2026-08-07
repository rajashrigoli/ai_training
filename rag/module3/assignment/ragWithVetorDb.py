import joblib
import weaviate
from weaviate.classes.query import (
    Filter, 
    Rerank
)

import os

from utils import (
    generate_with_single_input,
    print_object_properties,
    display_widget,
    kill_processes_on_ports
)
import test_grader as unittests

# Kill processes on ports before importing flask_app and weaviate_server
# WARNING: Running this cell twice may kill the active kernel
kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

import flask_app
import weaviate_server

dash_line = "=" * 100

print(dash_line)
print("2 - Setting up the Weaviate Client and loading the data")
print(dash_line)

client = weaviate.connect_to_local(port=8079, grpc_port=50050)
bbc_data = joblib.load('data/bbc_data.joblib')
print_object_properties(bbc_data[0])


print(dash_line)
print("3 - Loading the Collection")
print(dash_line)
collection = client.collections.get("bbc_collection")

object = collection.query.fetch_objects(limit = 1, include_vector = True).objects[0]
print("Printing the properties (some will be truncated due to size)")
print_object_properties(object.properties)
print("Vector: (truncated)",object.vector['main_vector'][0:15])
print("Vector length: ", len(object.vector['main_vector']))

print("\n\n")
print("3.1 Metadata filtering")
print(dash_line)

def filter_by_metadata(metadata_property: str, 
                       values: list[str], 
                       collection: "weaviate.collections.collection.sync.Collection" , 
                       limit: int = 5) -> list:
    """
    Retrieves objects from a specified collection based on metadata filtering criteria.

    This function queries a collection within the specified client to fetch objects that match 
    certain metadata criteria. It uses a filter to find objects whose specified 'property' contains 
    any of the given 'values'. The number of objects retrieved is limited by the 'limit' parameter.

    Args:
    metadata_property (str): The name of the metadata property to filter on.
    values (List[str]): A list of values to be matched against the specified property.
    collection_name (weaviate.collections.collection.sync.Collection): The collection to query.
    limit (int, optional): The maximum number of objects to retrieve. Defaults to 5.

    Returns:
    List[Object]: A list of objects from the collection that match the filtering criteria.
    """
    ### START CODE HERE ###
    
    # Retrieve using collection.query.fetch_objects
    
    response = collection.query.fetch_objects(
        limit=limit,
        filters=Filter.by_property(metadata_property).contains_any(values)
    )

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects


# Let's get an example
res = filter_by_metadata('title', ['Taylor Swift'], collection, limit = 2)
for x in res:
    print_object_properties(x)


# Test your solution!
unittests.test_filter_by_metadata(filter_by_metadata, client)

print("\n\n")
print("3.2 Semantic search")
print(dash_line)

def semantic_search_retrieve(query: str,
                             collection: "weaviate.collections.collection.sync.Collection" , 
                             top_k: int = 5) -> list:
    """
    Performs a semantic search on a collection and retrieves the top relevant chunks.

    This function executes a semantic search query on a specified collection to find text chunks 
    that are most relevant to the input 'query'. The search retrieves a limited number of top 
    matching objects, as specified by 'top_k'. The function returns the 'chunk' property of 
    each of the top matching objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the semantic search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    ### START CODE HERE ###

    # Retrieve using collection.query.near_text
    response = collection.query.near_text(query = query, limit = top_k)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects


print("\n\n")
print("3.3 BM25 Search")
print(dash_line)

# GRADED CELL 

def bm25_retrieve(query: str, 
                  collection: "weaviate.collections.collection.sync.Collection" , 
                  top_k: int = 5) -> list:
    """
    Performs a BM25 search on a collection and retrieves the top relevant chunks.

    This function executes a BM25-based search query on a specified collection to identify text 
    chunks that are most relevant to the provided 'query'. It retrieves a limited number of the 
    top matching objects, as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the BM25 search is performed.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    
    ### START CODE HERE ###

    # Retrieve using collection.query.bm25
    response = collection.query.bm25(
        query=query,
        limit=top_k
    )

    ### END CODE HERE ### 
    
    response_objects = [x.properties for x in response.objects]
    return response_objects 


print_object_properties(bm25_retrieve('Tell me about the last Taylor Swift show', collection, top_k = 2))

print("\n\n")
print("3.4 Hybrid search")
print(dash_line)

# GRADED CELL 

def hybrid_retrieve(query: str, 
                    collection: "weaviate.collections.collection.sync.Collection" , 
                    alpha: float = 0.5,
                    top_k: int = 5
                   ) -> list:
    """
    Performs a hybrid search on a collection and retrieves the top relevant chunks.

    This function executes a hybrid search that combines semantic vector search and traditional 
    keyword-based search on a specified collection to find text chunks most relevant to the 
    input 'query'. The relevance of results is influenced by 'alpha', which balances the weight 
    between vector and keyword matches. It retrieves a limited number of top matching objects, 
    as specified by 'top_k', and returns the 'chunk' property of these objects.

    Args:
    query (str): The search query used to find relevant text chunks.
    collection (weaviate.collections.collection.sync.Collection): The collection in which the hybrid search is performed.
    alpha (float, optional): A weighting factor that balances the contribution of semantic 
    and keyword matches. Defaults to 0.5.
    top_k (int, optional): The number of top relevant objects to retrieve. Defaults to 5.

    Returns:
    List[str]: A list of text chunks that are most relevant to the given query.
    """
    ### START CODE HERE ### 

    # Retrieve using collection.query.hybrid
    response = collection.query.hybrid(
        query=query,
        alpha=alpha,
        limit=top_k
    )

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 

print_object_properties(hybrid_retrieve('Tell me about the last Taylor Swift show', collection, top_k = 2))


print("\n\n")
print("3.5 Reranking search")
print(dash_line)

# GRADED CELL 

def semantic_search_with_reranking(query: str, 
                                   rerank_property: str,
                                   collection: "weaviate.collections.collection.sync.Collection" , 
                                   rerank_query: str = None,
                                   top_k: int = 5
                                   ) -> list:
    """
    Performs a semantic search and reranks the results based on a specified property.

    Args:
        query (str): The search query to perform the initial search.
        rerank_property (str): The property used for reranking the search results.
        collection (weaviate.collections.collection.sync.Collection): The collection to search within.
        rerank_query (str, optional): The query to use specifically for reranking. If not provided, 
                                      the original query is used for reranking.
        top_k (int, optional): The maximum number of top results to return. Defaults to 5.

    Returns:
        list: A list of properties from the reranked search results, where each item corresponds to 
              an object in the collection.
    """
    ### START CODE HERE ### 

    # Set the rerank_query to be the same as the query if rerank_query is not passed (don't change this line)
    if rerank_query is None: 
        rerank_query = query 
        
    # Define the reranker with rerank_query and rerank_property
    reranker = Rerank(query = rerank_query, prop = rerank_property)

    # Retrieve using collection.query.near_text with the appropriate parameters (do not forget the rerank!)
    response = collection.query.near_text(query = query, limit = top_k, rerank = reranker)

    ### END CODE HERE ###
    
    response_objects = [x.properties for x in response.objects]
    
    return response_objects 


# Set a query
query = 'Tell me about the conflicts in Latin America'
# Get the results from a search (in this case the hybrid search)
results = semantic_search_with_reranking(query, collection = collection, top_k = 2, rerank_property = 'chunk')

print_object_properties(results)

print("\n\n")
print(dash_line)
print("4 - Incorporating the Weaviate API into our previous schema")
print(dash_line)

print("\n\n")
print("4.1 Generating the final prompt")
print(dash_line)

def generate_final_prompt(query: str, 
                          top_k: int, 
                          retrieve_function: callable,
                          rerank_query: str = None, 
                          rerank_property: str = None, 
                          use_rerank: bool = False, 
                          use_rag: bool = True) -> str:
    """
    Generates a final prompt by optionally retrieving and formatting relevant documents using retrieval-augmented generation (RAG).

    Args:
        query (str): The initial query to be used for document retrieval.
        top_k (int): The number of top documents to retrieve and use for generating the prompt.
        retrieve_function (callable): The function used to retrieve documents based on the query.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.
        rerank_property (str, optional): The property used for reranking. Required if 'use_rerank' is True.
        use_rerank (bool, optional): Flag to denote whether to use reranking in document retrieval. Defaults to False.
        use_rag (bool, optional): Flag to determine whether to use retrieval-augmented generation. Defaults to True.

    Returns:
        str: A constructed prompt that includes the original query and formatted retrieved documents if 'use_rag' is True.
             Otherwise, it returns the original query.
    """
    # If no rag, return the query
    if not use_rag:
        return query
    
    if use_rerank:
        if rerank_property is None:
            raise ValueError('rerank_property must be set if use_rerank = True')
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection, rerank_property = rerank_property, rerank_query = rerank_query)
    else:
        top_k_documents = retrieve_function(query=query, top_k=top_k, collection = collection)
    
    # Initialize an empty string to store the formatted data.
    formatted_data = ""
    
    # Iterate over each retrieved document.
    for document in top_k_documents:
        # Format each document into a structured string.
        document_layout = (
            f"Title: {document['title']}, Chunk: {document['chunk']}, "
            f"Published at: {document['pubDate']}\nURL: {document['link']}"
        )
        # Append the formatted string to the main data string with a newline for separation.
        formatted_data += document_layout + "\n"
    
    # If use_rag flag is True, construct the enhanced prompt with the augmented data.
    retrieve_data_formatted = formatted_data  # Store formatted data.
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"The news data is ordered by relevance."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )
    
    return prompt

prompt = generate_final_prompt("Tell me the economic situation of the US in 2024.", top_k = 5, retrieve_function = semantic_search_retrieve, use_rerank = False, rerank_property = 'title')
print(prompt)

print("\n\n")
print("4.2 LLM call")
print(dash_line)

def llm_call(query: str, 
             retrieve_function: callable = None, 
             top_k: int = 5, 
             use_rag: bool = True, 
             use_rerank: bool = False, 
             rerank_property: str = None, 
             rerank_query: str = None) -> str:
    """
    Simulates a call to a language model by generating a prompt and using it to produce a response.

    Args:
        query (str): The initial query for which a response is sought.
        retrieve_function (callable, optional): The function used to retrieve documents related to the query.
        top_k (int, optional): The number of top documents to retrieve and use for generating the prompt. Defaults to 5.
        use_rag (bool, optional): Indicates whether to use retrieval-augmented generation. Defaults to True.
        use_rerank (bool, optional): Indicates whether to apply reranking to the retrieved documents. Defaults to False.
        rerank_property (str, optional): The property to use for reranking. Required if 'use_rerank' is True.
        rerank_query (str, optional): The query used specifically for reranking documents if reranking is enabled.

    Returns:
        str: The generated response content after processing the prompt with a language model.
    """
    
    # Get the prompt
    PROMPT = generate_final_prompt(query, top_k = top_k, retrieve_function = retrieve_function, use_rag = use_rag, use_rerank = use_rerank, rerank_property = rerank_property, rerank_query = rerank_query)
    
    generated_response = generate_with_single_input(PROMPT)

    generated_message = generated_response['content']
    
    return generated_message

query = "Tell me about United States and Brazil's relationship over the course of 2024. Provide links for the resources you use in the answer."

# Result with reranked results
print(llm_call(query = query, 
               top_k = 5, 
               retrieve_function = hybrid_retrieve, 
               ))

print("\n\n")
print(dash_line)
print("5 - Experimenting with Your RAG System")
print(dash_line)

display_widget(llm_call, semantic_search_retrieve, bm25_retrieve, hybrid_retrieve, semantic_search_with_reranking)