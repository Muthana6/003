from semantic_syntacic import extract_text_from_pdf  # Ensure this module exists and is correctly named
from Vector_Representation import get_word_vectors, aggregate_vectors
from similarity_measurement import calculate_cosine_similarity
from train_som import create_and_train_som
import numpy as np
from preprocessing_tokenization import preprocess_text
from Analizing_context import visualize_som_simple

def comparing_context(pdf_path1, pdf_path2, x_dim=5, y_dim=5):
    # Extract and preprocess text from the PDFs
    text1 = extract_text_from_pdf(pdf_path1)
    text2 = extract_text_from_pdf(pdf_path2)

    # Preprocess both texts
    words1, filtered_sentences1 = preprocess_text(text1)
    words2, filtered_sentences2 = preprocess_text(text2)

    # Vector representation for both documents
    word_vectors1 = get_word_vectors(words1)
    word_vectors2 = get_word_vectors(words2)
    document_vector1 = aggregate_vectors(word_vectors1)
    document_vector2 = aggregate_vectors(word_vectors2)

    # Calculate cosine similarity between the two document vectors
    context_cosine_similarity = calculate_cosine_similarity(document_vector1, document_vector2)
    print("Cosine Similarity between Text 1 and Text 2:", context_cosine_similarity)

    # Document vectors array for further processing (e.g., SOM training)
    data = np.array([document_vector1, document_vector2])
    labels = ["Text 1", "Text 2"]

    # Create and train the SOM with a fixed random seed for consistency
    som = create_and_train_som(data, x_size=x_dim, y_size=y_dim, input_len=data.shape[1], sigma=0.3, learning_rate=0.5, num_iteration=5000, random_seed=42)

    print("SOM Training Completed")

    # Retrieve and print the SOM weights
    som_weights = som.get_weights()
    print("SOM Weights:\n", som_weights)

    # Retrieve and print the winning nodes for the input vectors
    winning_nodes = [som.winner(vector) for vector in data]
    print("Winning Nodes for Input Vectors:", winning_nodes)

    # Visualize the SOM
    visualize_som_simple(som, data, labels, title='Document Positions on the Self-Organizing Map')

    return document_vector1, document_vector2

# Example usage:
if __name__ == "__main__":
    pdf_path1 = r'C:\Users\skykn\Downloads\Untitled document (22).pdf'
    pdf_path2 = r'C:\Users\skykn\Downloads\Untitled document (23).pdf'

    context_vectors = comparing_context(pdf_path1, pdf_path2)
