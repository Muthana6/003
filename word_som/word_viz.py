import numpy as np
import matplotlib.pyplot as plt
import fitz


def plot_som(som, feature_names, x, y):
    """
    Plot a Self-Organizing Map (SOM) showing the distances between neurons and labeling with top terms.

    Parameters:
    som (MiniSom): The trained SOM instance.
    feature_names (list): List of names corresponding to the features in the TF-IDF vectors.
    x, y (int): Dimensions of the SOM grid.
    """
    max_weight = np.max(som.get_weights())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    # Initialize a grid for neuron positions
    for position, weights in np.ndenumerate(som.get_weights()):
        # Position is a tuple (i, j)
        # weights is the neuron weight vector
        label = feature_names[np.argmax(weights)]
        ax.text(position[1], position[0], label, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.xlim([0, y])  # Note that MiniSom uses the second parameter as width (y-axis in plotting)
    plt.ylim([0, x])  # and the first parameter as height (x-axis in plotting)
    plt.title('SOM Grid with Most Representative Words per Neuron')
    plt.show()


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
