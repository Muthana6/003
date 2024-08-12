import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom  # Ensure you have MiniSom installed: pip install MiniSom

# Add the directory containing the word_som and context_som modules to the sys.path
 


def prepare_feature_vectors(combined_similarity_matrix):
    num_documents = combined_similarity_matrix.shape[0]
    feature_vectors = np.zeros((num_documents, num_documents))
    for i in range(num_documents):
        feature_vectors[i] = combined_similarity_matrix[i]

    # Replace NaN and infinite values with finite numbers
    feature_vectors = np.nan_to_num(feature_vectors)

    # Normalize the feature vectors
    min_val = np.min(feature_vectors, axis=0)
    ptp_val = np.ptp(feature_vectors, axis=0) + 1e-10  # Add small value to avoid division by zero
    feature_vectors = (feature_vectors - min_val) / ptp_val

    return feature_vectors


def train_som(feature_vectors, som_dimensions=(10, 10)):
    som = MiniSom(x=som_dimensions[0], y=som_dimensions[1], input_len=feature_vectors.shape[1], sigma=1.0,
                  learning_rate=0.5)
    som.random_weights_init(feature_vectors)
    som.train_random(data=feature_vectors, num_iteration=100)
    return som


def visualize_som(som, feature_vectors, som_dimensions=(10, 10)):
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(feature_vectors):
        w = som.winner(x)
        plt.text(w[0] + .5, w[1] + .5, str(i), color=plt.cm.jet(i / len(feature_vectors)),
                 fontdict={'weight': 'bold', 'size': 12})
    plt.xlim([0, som_dimensions[0]])
    plt.ylim([0, som_dimensions[1]])
    plt.title('Document SOM')
    plt.show()


def plot_umatrix(som, title='U-Matrix'):
    plt.figure(figsize=(10, 10))
    umatrix = som.distance_map().T  # transpose to match the SOM orientation
    umatrix = np.nan_to_num(umatrix)  # Ensure no NaN values
    umatrix_max = np.max(umatrix)
    if umatrix_max != 0:  # Prevent division by zero
        umatrix /= umatrix_max
    plt.imshow(umatrix, cmap='bone_r')
    plt.colorbar()
    plt.title(title)
    plt.show()


def main():
    pdf_path1 = r'C:\Users\skykn\Downloads\Untitled document (15).pdf'
    pdf_path2 = r'C:\Users\skykn\Downloads\Untitled document (16).pdf'

    pdf_files = [pdf_path1, pdf_path2]  # Add more PDF paths if needed
    num_files = len(pdf_files)

    word_similarity_matrix = np.zeros((num_files, num_files))
    context_similarity_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(num_files):
            word_similarity_matrix[i, j] = compare_word(pdf_files[i], pdf_files[j])
            context_similarity_matrix[i, j] = comparing_context(pdf_files[i], pdf_files[j])

    # Calculate combined similarity
    combined_similarity_matrix = calculate_combined_similarity(word_similarity_matrix, context_similarity_matrix)
    print("Combined Similarity Matrix:")
    print(combined_similarity_matrix)

    # Prepare feature vectors
    feature_vectors = prepare_feature_vectors(combined_similarity_matrix)
    print("Feature Vectors:")
    print(feature_vectors)

    # Train SOM
    som_dimensions = (10, 10)
    som = train_som(feature_vectors, som_dimensions)

    # Visualize SOM
    visualize_som(som, feature_vectors, som_dimensions)

    # U-Matrix Visualization
    plot_umatrix(som, title='SOM U-Matrix Visualization')


if __name__ == "__main__":
    main()
