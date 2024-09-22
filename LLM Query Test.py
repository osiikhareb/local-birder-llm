# -*- coding: utf-8 -*-
"""
LLM query test implementation with Ollama locally
Langchain will be used to integrate the local Ollama LLM of choice and the vector database

@author: Osi
"""

from langchain.embeddings import OpenAIEmbeddings  # Or use custom embeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import VectorDBQA

# Load the FAISS index
index = faiss.read_index('vector_index.faiss')

# Use FAISS vector store for LangChain
vectorstore = FAISS(embedding_function=None, index=index)

# Define a simple prompt for Ollama LLM
prompt_template = """Given the following context, answer the user's query.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Initialize the Ollama LLM
llm = Ollama(model="llama")  # Local LLM model from Ollama

# Combine LLM with FAISS Vector Store using LangChain's VectorDBQA
chain = VectorDBQA(llm=llm, vectorstore=vectorstore, prompt=prompt)

# Querying the database
question = "What bird is light brown with stripes on its back and a red patch on its head?"
result = chain.run(question)
print(result)