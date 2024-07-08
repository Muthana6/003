import matplotlib.pyplot as plt
import numpy as np


def plot_som_documents(som, tfidf_matrix, document_labels):
    """
    Visualize the trained SOM with document labels.

    Parameters:
    som (MiniSom): A trained SOM object.
    tfidf_matrix (ndarray): The TF-IDF matrix of the documents.
    document_labels (list): Labels for the documents.
    """
    x_dim, y_dim = som.get_weights().shape[:2]
    plt.figure(figsize=(10, 10))

    # Create an empty grid for document labels
    doc_grid = np.empty((x_dim, y_dim), dtype=object)
    for i in range(x_dim):
        for j in range(y_dim):
            doc_grid[i, j] = []

    # Assign each document to a position in the SOM
    for i, x in enumerate(tfidf_matrix):
        w = som.winner(x)
        doc_grid[w[0], w[1]].append(document_labels[i])

    # Plot the SOM with document labels
    for i in range(x_dim):
        for j in range(y_dim):
            if doc_grid[i, j]:  # If there is a document in this position
                plt.text(i + .5, j + .5, ', '.join(doc_grid[i, j]), fontsize=12, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.5, lw=0))

    plt.imshow(som.distance_map().T, cmap='bone_r', alpha=.5)
    plt.colorbar()
    plt.title('Self-Organizing Map (SOM) with Document Labels')
    plt.show()
