import joblib
import numpy as np
import bm25s
import os
from sentence_transformers import SentenceTransformer

from utils import (
    read_dataframe,
    pprint, 
    generate_with_single_input, 
    cosine_similarity,
    display_widget
)
import unittests

dash_line = '-' * 100

print("\n\n")
print(dash_line)
print("Loading the Dataset")
print(dash_line)

NEWS_DATA = read_dataframe("news_data_dedup.csv")
pprint(NEWS_DATA[5])

print("\n\n")
print(dash_line)
print("3 Retrieve Functions:")
print(dash_line)


print("\n\n")
print("3.1 Query news by index")
print(dash_line)
def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    dataset (list or sequence): The dataset from which elements are to be retrieved. It should support indexing.

    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
     
    output = [NEWS_DATA[index] for index in indices]

    return output

print("\n\n")
print("3.2 BM25 Retrieve")
print(dash_line)
# The corpus used will be the title appended with the description
corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]

# Instantiate the retriever by passing the corpus data
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)

# Tokenize the chunks
tokenized_data = bm25s.tokenize(corpus)

# Index the tokenized chunks within the retriever
BM25_RETRIEVER.index(tokenized_data)

# Tokenize the same query used in the previous exercise
sample_query = "What are the recent news about GDP?"
tokenized_sample_query = bm25s.tokenize(sample_query)

# Get the retrieved results and their respective scores
results, scores = BM25_RETRIEVER.retrieve(tokenized_sample_query, k=3)

print(f"Results for query: {sample_query}\n")
for doc in results[0]:
  print(f"Document retrieved {corpus.index(doc)} : {doc}\n")


# Use these as a global defined BM25 retriever objects

corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)
TOKENIZED_DATA = bm25s.tokenize(corpus)
BM25_RETRIEVER.index(TOKENIZED_DATA)


# GRADED CELL

def bm25_retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top k relevant documents for a given query using the BM25 algorithm.

    This function tokenizes the input query and uses a pre-indexed BM25 retriever to
    search through a collection of documents. It returns the indices of the top k documents
    that are most relevant to the query.

    Args:
        query (str): The search query for which documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k relevant documents
        within the corpus.
    """
    ### START CODE HERE ###

    # Tokenize the query using the 'tokenize' function from the 'bm25s' module
    tokenized_query = bm25s.tokenize(query)
    
    # Use the 'BM25_RETRIEVER' to retrieve documents and their scores based on the tokenized query
    # Retrieve the top 'k' documents
    results, scores = BM25_RETRIEVER.retrieve(tokenized_query, k=top_k)

    # Extract the first element from 'results' to get the list of retrieved documents
    results = results[0]

    # Convert the retrieved documents into their corresponding indices in the results list
    top_k_indices = [corpus.index(doc) for doc in results]

    ### END CODE HERE ###
    
    return top_k_indices


# Output is a list of indices
bm25_retrieve("What are the recent news about GDP?")

# Test your function!
unittests.test_bm25_retrieve(bm25_retrieve)

print("\n\n")
print("### 3.3 Semantic Search")
print(dash_line)

# Load the sentence transformer model
model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(os.path.join('', model_name))

# Load pre-computed embeddings
EMBEDDINGS = joblib.load("embeddings.joblib")

print("\n\n")
print("3.4 Retrieve news by Embedding")
print(dash_line)


print("\n\n")
print("Example of cosine similarity and embedding")
print(dash_line)

query1 = "What are the primary colors"
query2 = "Yellow, red and blue"
query3 = "Cats are friendly animals"

query1_embed = model.encode(query1)
query2_embed = model.encode(query2)
query3_embed = model.encode(query3)

print(f"Similarity between '{query1}' and '{query2}' = {cosine_similarity(query1_embed, query2_embed)[0]}")
print(f"Similarity between '{query1}' and '{query3}' = {cosine_similarity(query1_embed, query3_embed)[0]}")

print("\n\n")
print("Example with the full embedding")
print(dash_line)
query = "Taylor Swift"
query_embed = model.encode(query)
# The result is a matrix with one matrix per sample. Since there is only one sample (the query), it is a matrix with one matrix within.
# This is why you need to get the first element
similarity_scores = cosine_similarity(query_embed, EMBEDDINGS)
similarity_indices = np.argsort(-similarity_scores) # Sort on decreasing order (sort the negative on increasing order), but return the indices
# Top 2 indices
top_2_indices = similarity_indices[:2]
print(top_2_indices)

# Retrieving the data
query_news(top_2_indices)

print("\n\n")
print(dash_line)
print("Exercise 2: Semantic Search")
print(dash_line)

# GRADED CELL 

def semantic_search_retrieve(query, top_k=5):
    """
    Retrieves the top k relevant documents for a given query using semantic search and cosine similarity.

    This function generates an embedding for the input query and compares it against pre-computed document
    embeddings using cosine similarity. The indices of the top k most similar documents are returned.

    Args:
        query (str): The search query for which relevant documents need to be retrieved.
        top_k (int): The number of top relevant documents to retrieve. Default value is 5.

    Returns:
        List[int]: A list of indices corresponding to the top k most relevant documents in the corpus.
    """
    ### START CODE HERE ###
    # Generate the embedding for the query using the pre-trained model
    query_embedding = model.encode(query)
    
    # Calculate the cosine similarity scores between the query embedding and the pre-computed document embeddings
    similarity_scores = cosine_similarity(query_embedding, EMBEDDINGS)
    
    # Sort the similarity scores in descending order and get the indices
    similarity_indices = np.argsort(-similarity_scores)

    # Select the indices of the top k documents as a numpy array
    top_k_indices_array = similarity_indices[:top_k]

    ### END CODE HERE ###
    
    # Cast them to int 
    top_k_indices = [int(x) for x in top_k_indices_array]
    
    return top_k_indices

# Let's see an example
semantic_search_retrieve("What are the recent news about GDP?")

unittests.test_semantic_search_retrieve(semantic_search_retrieve, EMBEDDINGS)

print("\n\n")
print(dash_line)
print("3.6 RRF Retrieve")
print(dash_line)
print("\n\n")
print("Exercise 3: RRF Retrieve")
print(dash_line)

# GRADED CELL 
def reciprocal_rank_fusion(list1, list2, top_k=5, K=60):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
        list1 (list[int]): A list of indices of the top-k documents that match the query.
        list2 (list[int]): Another list of indices of the top-k documents that match the query.
        top_k (int): The number of top documents to consider from each list for fusion. Defaults to 5.
        K (int): A constant used in the RRF formula. Defaults to 60.

    Returns:
        list[int]: A list of indices of the top-k documents sorted by their RRF scores.
    """

    ### START CODE HERE ###

    # Create a dictionary to store the RRF scores for each document index
    rrf_scores = {}

    # Iterate over each document list
    for lst in [list1, list2]:
        # Calculate the RRF score for each document index
        for rank, item in enumerate(lst, start=1): # Start = 1 set the first element as 1 and not 0. 
                                                   # This is a convention on how ranks work (the first element in ranking is denoted by 1 and not 0 as in lists)
            # If the item is not in the dictionary, initialize its score to 0
            if item not in rrf_scores:
                rrf_scores[item] = 0
            # Update the RRF score for each document index using the formula 1 / (rank + K)
            rrf_scores[item] += 1 / (rank + K)

    # Sort the document indices based on their RRF scores in descending order
    sorted_items = sorted(rrf_scores.keys(), key=rrf_scores.get, reverse = True)

    # Slice the list to get the top-k document indices
    top_k_indices = [int(x) for x in sorted_items[:top_k]]

    ### END CODE HERE ###

    return top_k_indices

list1 = semantic_search_retrieve('What are the recent news about GDP?')
list2 = bm25_retrieve('What are the recent news about GDP?')
rrf_list = reciprocal_rank_fusion(list1, list2)
print(f"Semantic Search List: {list1}")
print(f"BM25 List: {list2}")
print(f"RRF List: {rrf_list}")

print("\n\n")
print(dash_line)
print("4 - Completing the RAG System")
print(dash_line)

print("\n\n")
print("4.1 Creating the final prompt")
print(dash_line)
def generate_final_prompt(query, top_k, retrieve_function = None, use_rag=True):
    """
    Generates an augmented prompt for a Retrieval-Augmented Generation (RAG) system by retrieving the top_k most 
    relevant documents based on a given query.

    Parameters:
    query (str): The search query for which the relevant documents are to be retrieved.
    top_k (int): The number of top relevant documents to retrieve.
    retrieve_function (callable): The function used to retrieve relevant documents. If 'reciprocal_rank_fusion', 
                                  it will combine results from different retrieval functions.
    use_rag (bool): A flag to determine whether to incorporate retrieved data into the prompt (default is True).

    Returns:
    str: A prompt that includes the top_k relevant documents formatted for use in a RAG system.
    """

    # Define the prompt as the initial query
    prompt = query
    
    # If not using rag, return the prompt
    if not use_rag:
        return prompt


    # Determine which retrieve function to use based on its name.
    if retrieve_function.__name__ == 'reciprocal_rank_fusion':
        # Retrieve top documents using two different methods.
        list1 = semantic_search_retrieve(query, top_k)
        list2 = bm25_retrieve(query, top_k)
        # Combine the results using reciprocal rank fusion.
        top_k_indices = retrieve_function(list1, list2, top_k)
    else:
        # Use the provided retrieval function.
        top_k_indices = retrieve_function(query=query, top_k=top_k)
    
    
    # Retrieve documents from the dataset using the indices.
    relevant_documents = query_news(top_k_indices)
    
    formatted_documents = []

    # Iterate over each retrieved document.
    for document in relevant_documents:
        # Format each document into a structured string.
        formatted_document = (
            f"Title: {document['title']}, Description: {document['description']}, "
            f"Published at: {document['published_at']}\nURL: {document['url']}"
        )
        # Append the formatted string to the main data string with a newline for separation.
        formatted_documents.append(formatted_document)

    retrieve_data_formatted = "\n".join(formatted_documents)
    
    prompt = (
        f"Answer the user query below. There will be provided additional information for you to compose your answer. "
        f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
        f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
        f"Query: {query}\n"
        f"2024 News: {retrieve_data_formatted}"
    )

    
    return prompt

def llm_call(query, retrieve_function = None, top_k = 5,use_rag = True):

    # Get the system and user dictionaries
    prompt = generate_final_prompt(query, top_k = top_k, retrieve_function = retrieve_function, use_rag = use_rag)

    generated_response = generate_with_single_input(prompt)

    generated_message = generated_response['content']
    
    return generated_message

query = "Recent news in technology. Provide sources."
print(llm_call(query, retrieve_function = semantic_search_retrieve))