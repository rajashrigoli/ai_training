from transformers.utils import logging
logging.set_verbosity_error()   

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
model = SentenceTransformer('all-MiniLM-L6-v2')

sentence1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']

embeddings1 = model.encode(sentence1)
print(embeddings1)

sentence2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The films are fantastic']
embeddings2 = model.encode(sentence2)
print(embeddings2)

cosine_scores = util.cos_sim(embeddings1, embeddings2)
print(cosine_scores)

for i in range(len(sentence1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentence1[i], sentence2[i], cosine_scores[i][i]))

