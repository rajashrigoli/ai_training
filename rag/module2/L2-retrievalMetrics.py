import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

dash_line = '-' * 100
print(dash_line)
print(" The dataset and the Embedding Model ")
print(dash_line)
from sklearn.datasets import fetch_20newsgroups

# Load the 20 Newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, data_home='./dataset')

# Convert the dataset to a DataFrame for easier handling
df = pd.DataFrame({
    'text': newsgroups_train.data,
    'category': newsgroups_train.target
})

# Display some basic information about the dataset
print(df.head())
print("\nDataset Size:", df.shape)
print("\nNumber of Categories:", len(newsgroups_train.target_names))
print("\nCategories:", newsgroups_train.target_names)

print(f"TEXT:\n\t{df['text'][0]}\nCATEGORY:\n\t{newsgroups_train.target_names[df['category'][0]]}")

print("\n\n")
print(dash_line)
print("Preprocessing and Vectorizing Data")
print(dash_line)
# Load the pre-trained sentence transformer model
model_name =  "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(os.path.join('',model_name))

embedding_vectors = joblib.load('embeddings.joblib')
print(len(embedding_vectors))

print("\n\n")
print(dash_line)
print("Basic functions for retrieval")
print(dash_line)

def preprocess_text(text):
    """
    Preprocess the text data by removing leading and trailing whitespace.

    Parameters:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text, with leading and trailing whitespace removed.
    """
    # Example preprocessing: remove leading/trailing whitespace
    text = text.strip()
    return text


def cosine_similarity(v1, array_of_vectors):
    """
    Cosine similarity between a vector and either a single vector (1D) or an array of vectors (2D).
    Returns a float for 1D input, or a list of floats for 2D input.
    Safely handles PyTorch tensors (moves to CPU) and NumPy arrays.
    """
    # Handle torch tensors for v1
    if hasattr(v1, "detach"):  # torch tensor
        v1 = v1.detach().cpu().numpy()
    v1 = np.asarray(v1, dtype=np.float32).ravel()

    # Handle torch tensors for array_of_vectors
    if hasattr(array_of_vectors, "detach"):  # torch tensor
        array_of_vectors = array_of_vectors.detach().cpu().numpy()
    A = np.asarray(array_of_vectors, dtype=np.float32)

    if A.ndim == 1:
        A = A.ravel()
        denom = np.linalg.norm(v1) * np.linalg.norm(A)
        return float(0.0 if denom == 0 else np.dot(v1, A) / denom)

    # 2D case: compute similarities for each row in A
    A = np.atleast_2d(A)
    v1_norm = np.linalg.norm(v1)
    A_norms = np.linalg.norm(A, axis=1)
    denom = v1_norm * A_norms
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = (A @ v1) / np.where(denom == 0, 1.0, denom)
    sims[denom == 0] = 0.0
    return sims.tolist()


def top_k_greatest_indices(lst, k):
    """
    Get the indices of the top k greatest items in a list.

    Parameters:
    lst (list): The list of elements to evaluate.
    k (int): The number of top elements to retrieve by index.

    Returns:
    list: A list of indices corresponding to the top k greatest elements in lst.
    """
    # Enumerate the list to keep track of indices
    indexed_list = list(enumerate(lst))
    # Sort by element values in descending order
    sorted_by_value = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # Extract the top k indices
    top_k_indices = [index for index, value in sorted_by_value[:k]]
    return top_k_indices

def retrieve_documents(query, embeddings, model, top_k=5):
    """
    Retrieve top-k most similar documents to a query using cosine similarity.
    Assumes:
      - preprocess_text, top_k_greatest_indices, df, and newsgroups_train are defined elsewhere.
      - embeddings is an iterable of document embeddings (NumPy arrays or torch tensors).
      - model.encode supports convert_to_tensor parameter (e.g., sentence-transformers).
    """
    
    query_clean = preprocess_text(query)
    query_embedding = model.encode(query_clean, convert_to_tensor=False).astype(np.float32)

    cosine_scores = []
    for x in embeddings:
        # Ensure each embedding is a NumPy array
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)

        score = cosine_similarity(query_embedding, x)  # returns a float for 1D x
        cosine_scores.append(float(score))

    top_results = top_k_greatest_indices(cosine_scores, k=top_k)

    print(f"Query: {query}")
    for idx in top_results:
        print(f"Document: {df.iloc[idx]['text'][:200]}...")
        print(f"Category: {newsgroups_train.target_names[df.iloc[idx]['category']]}...")
        print("\n\n")

        
# Example query
example_query = "space exploration"
retrieve_documents(example_query, embedding_vectors, model, top_k = 2)

print("\n\n")
print(dash_line)
print("Retrieving metrics")
print(dash_line)

def precision_at_k(relevant_count, k):
    """
    Calculate the Precision@K for a retrieval system.

    Precision@K is the ratio of relevant documents in the top K retrieved documents
    to the total number K of documents retrieved.

    Args:
        relevant_count (int): Number of relevant documents in the top K results.
        k (int): Total number of documents retrieved (K).

    Returns:
        float: The Precision@K value, or 0.0 if k is zero.
    
    Raises:
        ValueError: If any input is negative.
    """
    if relevant_count < 0 or k < 0:
        raise ValueError("All input values must be non-negative.")
    
    if k == 0:
        return 0.0

    return relevant_count / k

def recall_at_k(relevant_count, total_relevant):
    """
    Calculate the Recall@K for a retrieval system.

    Recall@K is the ratio of relevant documents in the top K retrieved documents
    to the total number of relevant documents in the entire corpus.

    Args:
        relevant_count (int): Number of relevant documents in the top K results.
        total_relevant (int): Total number of relevant documents in the corpus.

    Returns:
        float: The Recall@K value, or 0.0 if total_relevant is zero.
    
    Raises:
        ValueError: If any input is negative.
    """
    if relevant_count < 0 or total_relevant < 0:
        raise ValueError("All input values must be non-negative.")

    if total_relevant == 0:
        return 0.0

    return relevant_count / total_relevant

# Define more complex test queries with their corresponding desired categories
test_queries = [
    {"query": "advancements in space exploration technology", "desired_category": "sci.space"},
    {"query": "real-time rendering techniques in computer graphics", "desired_category": "comp.graphics"},
    {"query": "latest findings in cardiovascular medical research", "desired_category": "sci.med"},
    {"query": "NHL playoffs and team performance statistics", "desired_category": "rec.sport.hockey"},
    {"query": "impacts of cryptography in online security", "desired_category": "sci.crypt"},
    {"query": "the role of electronics in modern computing devices", "desired_category": "sci.electronics"},
    {"query": "motorcycles maintenance tips for enthusiasts", "desired_category": "rec.motorcycles"},
    {"query": "high-performance baseball tactics for championships", "desired_category": "rec.sport.baseball"},
    {"query": "historical influence of politics on society", "desired_category": "talk.politics.misc"},
    {"query": "latest technology trends in the Windows operating system", "desired_category": "comp.os.ms-windows.misc"}
    
]

def compute_metrics(queries, embeddings, model, top_k=5):
    """
    Compute Precision@K and Recall@K for a list of queries against a dataset of document embeddings.
    Assumes:
      - preprocess_text, top_k_greatest_indices, precision_at_k, recall_at_k, df, newsgroups_train are defined elsewhere.
      - embeddings is a list/iterable of document embeddings (NumPy arrays or torch tensors).
      - model.encode supports convert_to_tensor parameter (e.g., sentence-transformers).
    """

    results = []

    # Normalize all embeddings to NumPy once
    np_embeddings = []
    for x in embeddings:
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        np_embeddings.append(np.asarray(x, dtype=np.float32).ravel())
    E = np.vstack(np_embeddings)  # shape: (N, D)

    for item in queries:
        query = item["query"]
        desired_category = item["desired_category"]

        # Get NumPy, not torch, to avoid GPU->NumPy conversion errors
        q_clean = preprocess_text(query)
        q_emb = model.encode(q_clean, convert_to_tensor=False)
        q_emb = np.asarray(q_emb, dtype=np.float32).ravel()

        # Compute similarities vectorized
        cosine_scores = cosine_similarity(q_emb, E)  # list of floats length N

        # Top-K indices
        top_results = top_k_greatest_indices(cosine_scores, k=top_k)

        # Retrieved categories
        retrieved_categories = [
            newsgroups_train.target_names[df.iloc[idx]["category"]] for idx in top_results
        ]

        # Metrics
        relevant_in_top_k = sum(1 for cat in retrieved_categories if cat == desired_category)
        total_relevant_in_corpus = sum(
            1 for idx in range(len(df))
            if newsgroups_train.target_names[df.iloc[idx]["category"]] == desired_category
        )

        p = precision_at_k(relevant_in_top_k, top_k)
        r = recall_at_k(relevant_in_top_k, total_relevant_in_corpus)

        results.append({
            "query": query,
            "precision@k": p,
            "recall@k": r,
        })

    return results


# Run the queries and compute metrics with different K values
k_values = [5, 20, 50]

for k in k_values:
    print(f"\n{'='*80}")
    print(f"Results with K={k}:")
    print('='*80)
    results = compute_metrics(test_queries, embedding_vectors, model, top_k=k)
    
    # Display the results
    for result in results:
        print(f"Query: {result['query']}")
        print(f"  Precision@{k}: {result['precision@k']:.2f}, Recall@{k}: {result['recall@k']:.2f}")
        print()