import matplotlib.pyplot as plt
import numpy as np

def visualize_som_simple(som, data, labels, title='SOM Visualization'):
    """
    Visualize the SOM grid with the given data and labels.

    Parameters:
    som (MiniSom): The trained SOM.
    data (ndarray): The document vectors used to train the SOM.
    labels (list): The labels corresponding to the document vectors.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(data):
        w = som.winner(x)
        plt.text(w[0] + 0.5, w[1] + 0.5, labels[i], ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.xlim([0, som.get_weights().shape[0]])
    plt.ylim([0, som.get_weights().shape[1]])
    plt.grid()
    plt.title(title)
    plt.show()
