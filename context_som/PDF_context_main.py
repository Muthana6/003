from semantic_syntacic import extract_text_from_pdf
from semantic_syntacic import semantic_syntactic_analysis
from Vector_Representation import get_word_vectors, aggregate_vectors
from similarity_measurement import calculate_cosine_similarity
from train_som import create_and_train_som
import numpy as np
from preprocessing_tokenization import preprocess_text

import matplotlib.pyplot as plt
from Analizing_context import visualize_som_simple


def comparing_context(pdf_paths, x_dim=5, y_dim=5):
    # Initialize lists to hold data
    document_vectors = []
    labels = []

    # Process each PDF file
    for i, pdf_path in enumerate(pdf_paths):
        # Extract and preprocess text from the PDF
        text = extract_text_from_pdf(pdf_path)

        # Preprocess text
        words, filtered_sentences = preprocess_text(text)

        # Vector representation for the document
        word_vectors = get_word_vectors(words)
        document_vector = aggregate_vectors(word_vectors)

        # Store document vector and label
        document_vectors.append(document_vector)
        labels.append(f"Text {i + 1}")

    # Convert list to NumPy array for SOM training
    data = np.array(document_vectors)

    # Calculate cosine similarities between all pairs of documents
    cosine_similarities = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i, len(data)):
            similarity = calculate_cosine_similarity(data[i], data[j])
            cosine_similarities[i][j] = similarity
            cosine_similarities[j][i] = similarity
            if i != j:
                print(f"Cosine Similarity between Text {i + 1} and Text {j + 1}: {similarity}")

    # Create and train the SOM with a fixed random seed for consistency
    som = create_and_train_som(data, x_size=x_dim, y_size=y_dim, input_len=data.shape[1], sigma=0.3, learning_rate=0.5,
                               num_iteration=5000, random_seed=42)

    print("SOM Training Completed")

    # Retrieve and print the SOM weights
    som_weights = som.get_weights()
    print("SOM Weights:\n", som_weights)

    # Retrieve and print the winning nodes for the input vectors
    winning_nodes = [som.winner(vector) for vector in data]
    print("Winning Nodes for Input Vectors:", winning_nodes)

    # Visualize the SOM
    visualize_som_simple(som, data, labels, title='Document Positions on the Self-Organizing Map')

    return cosine_similarities


# Example usage:
pdf_paths = [
    r'../pdf data/pdf_2.pdf',
    r'../pdf data/pdf_3.pdf',
    r'../pdf data/pdf_33 .pdf'
]

cosine_similarities = comparing_context(pdf_paths)
