import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def analyze_clusters(som, data, labels):
    """
    Analyze the clusters formed by the SOM.

    Parameters:
    som (MiniSom): The trained SOM.
    data (ndarray): The document vectors used to train the SOM.
    labels (list): The labels corresponding to the document vectors.

    Returns:
    dict: A dictionary with the cluster assignments.
    """
    cluster_assignments = {}
    for i, vector in enumerate(data):
        winning_node = som.winner(vector)
        if winning_node not in cluster_assignments:
            cluster_assignments[winning_node] = []
        cluster_assignments[winning_node].append(labels[i])

    return cluster_assignments

def visualize_som(som, data, labels, title='SOM Visualization'):
    """
    Visualize the SOM grid.

    Parameters:
    som (MiniSom): The trained SOM.
    data (ndarray): The document vectors used to train the SOM.
    labels (list): The labels corresponding to the document vectors.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 10))
    for i, (x, t) in enumerate(zip(data, labels)):
        w = som.winner(x)
        plt.text(w[0] + 0.5, w[1] + 0.5, t, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.xlim([0, som.get_weights().shape[0]])
    plt.ylim([0, som.get_weights().shape[1]])
    plt.grid()
    plt.title(title)
    plt.show()

def visualize_clusters(som, cluster_assignments, title='SOM Cluster Visualization'):
    """
    Visualize the clusters on the SOM grid.

    Parameters:
    som (MiniSom): The trained SOM.
    cluster_assignments (dict): The cluster assignments.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 10))
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['r', 'g', 'b', 'c', 'm']

    for i, (node, texts) in enumerate(cluster_assignments.items()):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        for j, text in enumerate(texts):
            plt.plot(node[0] + 0.5, node[1] + 0.5, marker + color, markersize=12)
            plt.text(node[0] + 0.5, node[1] + 0.5 + j * 0.3, text, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, lw=0))

    patches = [mpatches.Patch(color=colors[i % len(colors)], label=f'Cluster {i + 1}') for i in
               range(len(cluster_assignments))]
    plt.legend(handles=patches)

    plt.xlim([0, som.get_weights().shape[0]])
    plt.ylim([0, som.get_weights().shape[1]])
    plt.grid()
    plt.title(title)
    plt.show()
