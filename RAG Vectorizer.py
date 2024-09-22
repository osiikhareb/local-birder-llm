# -*- coding: utf-8 -*-
"""
The text from the species descriptions will be cleaned, vectorized, and stored in a vector database for the purpose of retrieval augmented generation.
FAISS will be the vector database of choice since it can be run locally

@author: Osi
"""


import re
import nltk
import faiss
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



# Preprocess species descriptions

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

cleaned_text = preprocess(text)
print(cleaned_text)



# Vectorize the text

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([cleaned_text])
print(vectors.toarray())



# Store vectorized text in the FAISS database

# Assume vectors are stored in a list
vectors = np.array(vectors_list).astype('float32')
ids = np.array([i for i in range(len(vectors))])

# Build the index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)  # Add vectors to index

# Save the index
faiss.write_index(index, 'vector_index.faiss')

# To load the index later
index = faiss.read_index('vector_index.faiss')

# Query the index
query_vector = vectors[0]
k = 5  # Number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_vector]), k)
print(indices, distances)


