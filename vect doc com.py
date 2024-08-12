from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from context_som.comparing_contex import pdf_path1,pdf_path2
from context_som.PDF_context_main import visualize_som_simple
from word_som.compare_word import pdf_path1,pdf_path2
from word_som import word_viz,word_viz2,word_som_train


def calculate_combined_vectors(word_vectors, context_vectors):
    combined_vectors = []
    for word_vec, context_vec in zip(word_vectors, context_vectors):
        combined_vector = (word_vec + context_vec) / 2
        combined_vectors.append(combined_vector)
    return np.array(combined_vectors)

# Get word vectors and context vectors
word_vectors = compare_word(pdf_path1, pdf_path2)
context_vectors1, context_vectors2 = comparing_context(pdf_path1, pdf_path2)

# Ensure word vectors and context vectors have the same shape
word_vectors = np.array(word_vectors)
context_vectors = np.array([context_vectors1, context_vectors2])

# Calculate combined vectors
combined_vectors = calculate_combined_vectors(word_vectors, context_vectors)

# Train SOM on combined vectors
x_dim, y_dim = 5, 5
som = create_and_train_som(combined_vectors, x_size=x_dim, y_size=y_dim, input_len=combined_vectors.shape[1], sigma=0.3, learning_rate=0.5, num_iteration=5000, random_seed=42)

# Define document labels for visualization purposes
document_labels = ['Document 1', 'Document 2']

# Visualize the SOM with document labels
visualize_som_simple(som, combined_vectors, document_labels, title='Document Positions on the Self-Organizing Map')

# Print the cosine similarity between combined document vectors
cosine_combined_similarity = cosine_similarity(combined_vectors)
print('Cosine similarity between Document 1 and Document 2:', cosine_combined_similarity[0][1])
