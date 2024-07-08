import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Sample TF-IDF matrix
tfidf_matrix = np.array([
    [0.1, 0.3, 0.5, 0.0],
    [0.2, 0.2, 0.6, 0.0],
    [0.4, 0.2, 0.4, 0.0],
    [0.7, 0.1, 0.1, 0.1]
])

# Example document labels
documents = ['Doc1', 'Doc2', 'Doc3', 'Doc4']

# Calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# Create a graph
G = nx.Graph()

# Add nodes
for i, doc in enumerate(documents):
    G.add_node(i, label=doc)

# Add edges with weights
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        G.add_edge(i, j, weight=cosine_sim_matrix[i, j])

# Draw the graph
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.title('Cosine Similarity Graph')
plt.show()
