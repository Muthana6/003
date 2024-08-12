import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK data (only needs to be run once)
nltk.download('punkt')

# Example list of documents (each document is a string)
raw_documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Tokenize each document into words
document_list = [word_tokenize(doc.lower()) for doc in raw_documents]

# Create TaggedDocument objects for Doc2Vec
tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(document_list)]

# Initialize and train the Doc2Vec model
model = Doc2Vec(tagged_documents, vector_size=300, window=2, min_count=1, workers=4)

# Infer vector for a new document
new_doc = word_tokenize("This is a new document.".lower())
new_doc_vector = model.infer_vector(new_doc)

print(new_doc_vector)
