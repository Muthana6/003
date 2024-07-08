import numpy as np
import spacy

# Load the medium Spacy model with GloVe vectors
nlp = spacy.load('en_core_web_md')  # Ensure to download this model first


def get_word_vectors(words):
    """
    Generate vectors for a list of words using pre-trained GloVe embeddings.

    Parameters:
    words (list): List of words to vectorize.

    Returns:
    list: List of word vectors.
    """
    vectors = [nlp(word).vector for word in words if nlp(word).has_vector]
    return vectors


def aggregate_vectors(vectors):
    """
    Aggregate a list of word vectors into a single document vector using averaging.

    Parameters:
    vectors (list): List of word vectors.

    Returns:
    ndarray: A single aggregated document vector.
    """
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros((nlp.meta['vectors']['width'],))  # Return a zero vector if no valid vectors are available

