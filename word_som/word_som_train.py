# create_and_train_som.py
from minisom import MiniSom
import numpy as np


def create_and_train_som(tfidf_matrix, x=5, y=5, input_len=None, sigma=1.0, learning_rate=0.5, num_iteration=100):
    """
    Create and train a Self-Organizing Map (SOM) using the given TF-IDF matrix.

    Parameters:
    tfidf_matrix (ndarray): An array of TF-IDF vectors.
    x, y (int): Dimensions of the SOM grid.
    input_len (int): The number of features in the TF-IDF vectors.
    sigma (float): Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
    learning_rate (float): Learning rate of the SOM.
    num_iteration (int): Number of iterations for SOM training.

    Returns:
    MiniSom: A trained SOM object.
    """
    # Ensure tfidf_matrix is not empty
    if tfidf_matrix.size == 0:
        raise ValueError("TF-IDF matrix is empty.")

    # Set input_len to the number of features in tfidf_matrix if not provided
    if input_len is None:
        input_len = tfidf_matrix.shape[1]

    # Ensure input_len matches the number of features in tfidf_matrix
    if input_len != tfidf_matrix.shape[1]:
        raise ValueError(
            f"input_len ({input_len}) does not match the number of features in tfidf_matrix ({tfidf_matrix.shape[1]}).")

    # Initialize and train the SOM
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(tfidf_matrix)
    som.train_random(tfidf_matrix, num_iteration=num_iteration)

    return som

