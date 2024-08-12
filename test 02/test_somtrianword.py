import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Generate base vectors for high similarity groups
base_vector_1 = np.random.rand(60) * 0.1  # Base for Doc1 and Doc2
base_vector_2 = np.random.rand(60) * 0.1  # Base for Doc3 and Doc4
distinct_vector = np.random.rand(60)      # Distinct vector for Doc5

# Create vectors with 99% similarity
doc1_vector = base_vector_1 + np.random.rand(60) * 0.01
doc2_vector = base_vector_1 + np.random.rand(60) * 0.01
doc3_vector = base_vector_2 + np.random.rand(60) * 0.01
doc4_vector = base_vector_2 + np.random.rand(60) * 0.01
doc5_vector = distinct_vector

# Combine all vectors into one array
tfidf_results = np.array([doc1_vector, doc2_vector, doc3_vector, doc4_vector, doc5_vector])

# Train SOM
x_dim, y_dim = 5, 5
som = MiniSom(x=x_dim, y=y_dim, input_len=tfidf_results.shape[1], sigma=0.5, learning_rate=0.3)
som.random_weights_init(tfidf_results)
som.train_random(tfidf_results, num_iteration=100)

# Visualize document positions on SOM
def plot_document_positions(som, data, labels):
    plt.figure(figsize=(10, 10))
    for i, doc in enumerate(data):
        w = som.winner(doc)
        plt.text(w[0], w[1], labels[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.xticks(np.arange(som.get_weights().shape[0]))
    plt.yticks(np.arange(som.get_weights().shape[1]))
    plt.grid()
    plt.title('Document Positions on the Self-Organizing Map')
    plt.show()

# Define document labels for visualization purposes
document_labels = ['Doc1', 'Doc2', 'Doc3', 'Doc4', 'Doc5']

# Visualize document positions on SOM
plot_document_positions(som, tfidf_results, document_labels)
