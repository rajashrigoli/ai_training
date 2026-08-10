#!/usr/bin/env python
# coding: utf-8

# # Question Answering

# ## Overview
# 
# Recall the overall workflow for retrieval augmented generation (RAG):

# ![overview.jpeg](attachment:overview.jpeg)

# We discussed `Document Loading` and `Splitting` as well as `Storage` and `Retrieval`.
# 
# Let's load our vectorDB. 

import os
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# The code below was added to assign the openai LLM version filmed until it is deprecated, currently in Sept 2023. 
# LLM responses can often vary, but the responses may be significantly different when using a different model version.

llm_name = "llama3-8b-8192"
print(llm_name)


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
persist_directory = 'docs/chroma/'
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


print(vectordb._collection.count())


question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)


from langchain_groq import ChatGroq
llm = ChatGroq(model_name=llm_name, temperature=0)


# ### RetrievalQA chain

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)


result = qa_chain({"query": question})
print(result["result"])


# ### Prompt

# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


question = "Is probability a class topic?"


result = qa_chain({"query": question})
print(result["result"])
print(result["source_documents"][0])


# ### RetrievalQA chain types

qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)


result = qa_chain_mr({"query": question})
print(result["result"])


# If you wish to experiment on the `LangSmith platform` (previously known as LangChain Plus):
# 
#  * Go to [LangSmith](https://www.langchain.com/langsmith) and sign up
#  * Create an API key from your account's settings
#  * Use this API key in the code below   
#  * uncomment the code  
#  Note, the endpoint in the video differs from the one below. Use the one below.

#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
#os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key


qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
print(result["result"])


qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
print(result["result"])


# ### RetrievalQA limitations
#  
# QA fails to preserve conversational history.

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)


question = "Is probability a class topic?"
result = qa_chain({"query": question})
print(result["result"])


question = "why are those prerequesites needed?"
result = qa_chain({"query": question})
print(result["result"])


# Note, The LLM response varies. Some responses **do** include a reference to probability which might be gleaned from referenced documents. The point is simply that the model does not have access to past questions or answers, this will be covered in the next section.



