from minisom import MiniSom
import numpy as np

def create_and_train_som(data, x_size=5, y_size=5, input_len=300, sigma=0.3, learning_rate=0.5, num_iteration=5000, random_seed=42):
    """
    Create and train a Self-Organizing Map (SOM) with the given document vectors.

    Parameters:
    data (ndarray): The document vectors to train the SOM.
    x_size, y_size (int): Dimensions of the SOM.
    input_len (int): Dimensionality of the document vectors.
    sigma (float): Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
    learning_rate (float): Initial learning rate for SOM training.
    num_iteration (int): Number of iterations for training.
    random_seed (int): Seed for the random number generator for reproducibility.

    Returns:
    MiniSom: The trained SOM.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_seed)

    # Initialize the SOM
    som = MiniSom(x_size, y_size, input_len, sigma=sigma, learning_rate=learning_rate)

    # Initialize the weights randomly
    som.random_weights_init(data)

    # Train the SOM
    som.train_random(data, num_iteration)

    return som
