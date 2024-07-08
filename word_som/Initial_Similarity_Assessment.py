from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(tfidf_matrix):
    """
    Compute the cosine similarity between the first document vector and all other document vectors.

    Parameters:
    tfidf_matrix (sparse matrix): The TF-IDF matrix.

    Returns:
    similarity_list: A list with similarity scores of the first document to all other documents.
    """
    # Compute the cosine similarity matrix from the TF-IDF matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Extract the similarity scores for the first document
    similarity_list = [similarity_matrix[0][j] for j in range(len(similarity_matrix[0])) if 0 != j]

    return similarity_list




#def compute_similarity
