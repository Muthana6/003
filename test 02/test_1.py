import matplotlib.pyplot as plt
import numpy as np

def plot_som(som, feature_names, x_dim, y_dim):
    """
    Visualize the Self-Organizing Map (SOM).

    Parameters:
    som (MiniSom): Trained SOM object.
    feature_names (list): List of feature names corresponding to the input vectors.
    x_dim (int): Width of the SOM grid.
    y_dim (int): Height of the SOM grid.
    """
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(som.get_weights()):
        for j, y in enumerate(x):
            plt.text(i, j, feature_names[np.argmax(y)], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.xticks(np.arange(x_dim))
    plt.yticks(np.arange(y_dim))
    plt.grid()
    plt.title('Self-Organizing Map of TF-IDF Features')
    plt.show()

def plot_document_positions(som, data, labels):
    """
    Plot the positions of documents on the SOM.

    Parameters:
    som (MiniSom): Trained SOM object.
    data (ndarray): TF-IDF matrix of the documents.
    labels (list): Labels for the documents.
    """
    plt.figure(figsize=(10, 10))
    for i, doc in enumerate(data):
        w = som.winner(doc)
        plt.text(w[0], w[1], labels[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.xticks(np.arange(som.get_weights().shape[0]))
    plt.yticks(np.arange(som.get_weights().shape[1]))
    plt.grid()
    plt.title('Document Positions on the Self-Organizing Map')
    plt.show()
