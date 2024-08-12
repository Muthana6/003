import sys
import os
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Add the directory containing the word_som and context_som modules to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'word_som'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'context_som'))

from word_som.PDF_word_main import compare_word
from context_som.PDF_context_main import comparing_context

def calculate_combined_similarity(word_similarity, context_similarity, word_weight=0.5, context_weight=0.5):
    combined_similarity = (word_weight * word_similarity) + (context_weight * context_similarity)
    return combined_similarity

def create_and_train_som(data, x=5, y=5, input_len=None, sigma=1.0, learning_rate=0.5):
    if input_len is None:
        input_len = data.shape[1]
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data)
    som.train_random(data, num_iteration=100)
    return som

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

# File paths
pdf_path1 = r'C:\Users\skykn\Downloads\main.pdf'
pdf_path2 = r'C:\Users\skykn\Downloads\duc1.pdf'
pdf_path3 = r'C:\Users\skykn\Downloads\duc2.pdf'
pdf_path4 = r'C:\Users\skykn\Downloads\Untitled document (18).pdf'
pdf_path5 = r'C:\Users\skykn\Downloads\Untitled document (19).pdf'

# List of PDF files to compare
pdf_files = [pdf_path2, pdf_path2, pdf_path2, pdf_path2]
results = []

# Calculate combined similarities
for file_path in pdf_files:
    word_cosine = compare_word(pdf_path1, file_path)
    context_cosine = comparing_context(pdf_path1, file_path)
    combined_similarity = calculate_combined_similarity(word_cosine, context_cosine)
    results.append((os.path.basename(file_path), combined_similarity))

# Prepare data for SOM
combined_similarity_values = [result[1] for result in results]
combined_similarity_matrix = np.array(combined_similarity_values).reshape(-1, 1)

# Train SOM
x_dim, y_dim = 5, 5
som = create_and_train_som(combined_similarity_matrix, x=x_dim, y=y_dim)

# Visualize document positions on SOM
document_labels = [result[0] for result in results]
plot_document_positions(som, combined_similarity_matrix, document_labels)
