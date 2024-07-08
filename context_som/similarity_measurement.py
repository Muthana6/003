from scipy.spatial.distance import cosine

def calculate_cosine_similarity(vector1, vector2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vector1, vector2 (ndarray): Two vectors between which to calculate the cosine similarity.

    Returns:
    float: Cosine similarity between the two vectors.
    """
    # Compute cosine similarity as 1 - cosine distance
    similarity = 1 - cosine(vector1, vector2)
    return similarity
