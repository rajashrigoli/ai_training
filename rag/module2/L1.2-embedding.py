
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances as euclidean_distance
import numpy as np
from sentence_transformers import SentenceTransformer

print("The Embedding Model")

# Load the pre-trained sentence transformer model
model_name =  "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(os.path.join('',model_name))
# To get a string embedded, just pass it to the model.
res = model.encode("RAG is awesome")
print(res.shape)

# An array of strings can be passed, and the output will be an array of vectors, each with 768 dimensions.
model.encode(['apple', 'car'])

# Print the first 100 elements of the embedding.
print(res[:100])

print("Embeddings in Practice")
words = ['apple', 'car', 'fruit', 'automobile', 'love', 'sentiment']
vectorized_words = model.encode(words)

word = 'apple'
print(f"{word}:")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)].reshape(1, -1)
    print(f"\t{w}:\t\tCosine Similarity: {cosine_similarity(vectorized_word, vectorized_words[i].reshape(1, -1))[0][0]:.4f}")
print("\n\n\n")
for i, w in enumerate(words):
    # Get the vectorized word for the word defined above
    vectorized_word = vectorized_words[words.index(word)].reshape(1, -1)
    print(f"\t{w}:\t\tEuclidean Distance: {euclidean_distance(vectorized_word, vectorized_words[i].reshape(1, -1))[0][0]:.4f}")


def retrieve_relevant(query, documents, metric='cosine_similarity'):
    """
    Retrieves and ranks documents based on their similarity to a given query using the specified metric.
    
    Parameters:
    query (str): The query string for which relevant documents are to be retrieved.
    documents (list of str): A list of documents to be compared against the query.
    metric (str, optional): The similarity measurement metric to be used. It supports 'cosine_similarity'
                            and 'euclidean'. Defaults to 'cosine_similarity'.
    
    Returns:
    list of tuples: A list of tuples where each tuple contains a document and its similarity or distance
                    score with respect to the query. The list is sorted based on these scores, with
                    descending order for 'cosine_similarity' and ascending order for 'euclidean'.
    """
    query_emb = model.encode(query).reshape(1, -1)
    documents_emb = model.encode(documents)
    vals = []

    if metric == 'cosine_similarity':
        distances = cosine_similarity(query_emb, documents_emb)[0]
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in descending order
        vals.sort(reverse=True, key=lambda x: x[1])
        
    elif metric == 'euclidean':
        distances = euclidean_distance(query_emb, documents_emb)[0]
        vals = [(doc, dist) for doc, dist in zip(documents, distances)]
        # Sort in ascending order
        vals.sort(key=lambda x: x[1])
        
    return vals


documents = [
    "Mt. Fuji is a breathtaking place to explore during autumn.",
    "Santorini offers stunning views to admire during spring.",
    "Banff National Park is a picturesque destination to visit in the summer.",
    "The Great Wall of China is a spectacular site to experience during winter.",
    "The fjords of Norway are a magical place to cruise through in the spring.",
    "Prague is an enchanting city to wander through in winter.",
    "Kyoto's cherry blossoms create a beautiful scene to witness during spring.",
    "Marrakech offers vibrant markets and culture to enjoy in the fall.",
    "The Maldives are a paradisiacal getaway to savor during summer.",
    "The Christmas markets in Vienna are a festive delight to explore in winter."
]
print("Cosine Similarity Retrieval:")
query = "Suggest to me great places to visit in Asia."
score = retrieve_relevant(query, documents, metric='cosine_similarity')
print(score)

print("Euclidean Distance Retrieval:")
score = retrieve_relevant(query, documents, metric='euclidean')
print(score)

print("Embeddings and Input Size")
big_text = open("large_text.txt").read()
print(len(big_text))

print("Embedding the entire text:")
# Entire text
big_text_embedding = model.encode(big_text)

# Text with fewer characters
big_text_embedding_few_characters = model.encode(big_text[:3000])

# Checking if they are the same
print(np.array_equal(big_text_embedding, big_text_embedding_few_characters))