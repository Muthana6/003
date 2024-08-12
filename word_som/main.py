from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Example documents (already processed and vectorized)
doc1 = "quick brown fox jump lazy dog dog bark loudly"
doc2 = "fox sleep shade tree"

# Preprocessed and vectorized data (for simplicity, assume these are the TF-IDF vectors)
tfidf_matrix = np.array([[0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124],
                         [0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0.0]])

# Define SOM dimensions and parameters
x_dim = 5
y_dim = 5
input_len = tfidf_matrix.shape[1]
sigma = 0.5
learning_rate = 0.5
num_iteration = 100

# Initialize and train the SOM
som = MiniSom(x=x_dim, y=y_dim, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
som.random_weights_init(tfidf_matrix)
som.train_random(tfidf_matrix, num_iteration)

# Visualize the documents on the SOM grid
plt.figure(figsize=(8, 8))
for i, doc_vector in enumerate(tfidf_matrix):
    winner = som.winner(doc_vector)
    plt.text(winner[0] + 0.5, winner[1] + 0.5, f'Doc{i+1}', ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.xlim([0, x_dim])
plt.ylim([0, y_dim])
plt.grid()
plt.title('Document Positions on the Self-Organizing Map')
plt.show()
