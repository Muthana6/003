
# testing.....................


# from semantic_syntacic import extract_text_from_pdf
# from semantic_syntacic import semantic_syntactic_analysis
# from Vector_Representation import get_word_vectors, aggregate_vectors
# from similarity_measurement import calculate_cosine_similarity
# from train_som import create_and_train_som
# import numpy as np
# from preprocessing_tokenization import preprocess_text
#
# import matplotlib.pyplot as plt
# from Analizing_context import analyze_clusters, visualize_som, visualize_clusters
#
# def plot_umatrix(som, title='U-Matrix'):
#     plt.figure(figsize=(10, 10))
#     umatrix = som.distance_map().T  # transpose to match the SOM orientation
#     plt.imshow(umatrix, cmap='bone_r')
#     plt.colorbar()
#     plt.title(title)
#     plt.show()
#
# def comparing_context(pdf_path1, pdf_path2, x_dim=5, y_dim=5):
#     # Extract and preprocess text from the PDFs
#     text1 = extract_text_from_pdf(pdf_path1)
#     text2 = extract_text_from_pdf(pdf_path2)
#
#     # Preprocess both texts
#     words1, filtered_sentences1 = preprocess_text(text1)
#     words2, filtered_sentences2 = preprocess_text(text2)
#
#     # Vector representation for both documents
#     word_vectors1 = get_word_vectors(words1)
#     word_vectors2 = get_word_vectors(words2)
#     document_vector1 = aggregate_vectors(word_vectors1)
#     document_vector2 = aggregate_vectors(word_vectors2)
#
#     # Calculate cosine similarity between the two document vectors
#     context_cosine_similarity = calculate_cosine_similarity(document_vector1, document_vector2)
#     print("Cosine Similarity between Text 1 and Text 2:", context_cosine_similarity)
#
#     # Document vectors array for further processing (e.g., SOM training)
#     data = np.array([document_vector1, document_vector2])
#     labels = ["Text 1", "Text 2"]
#
#     # Create and train the SOM with a fixed random seed for consistency
#     som = create_and_train_som(data, x_size=x_dim, y_size=y_dim, input_len=data.shape[1], sigma=0.3, learning_rate=0.5, num_iteration=5000, random_seed=42)
#
#     print("SOM Training Completed")
#
#     # Retrieve and print the SOM weights
#     som_weights = som.get_weights()
#     print("SOM Weights:\n", som_weights)
#
#     # Retrieve and print the winning nodes for the input vectors
#     winning_nodes = [som.winner(vector) for vector in data]
#     print("Winning Nodes for Input Vectors:", winning_nodes)
#
#     # Analyze clusters
#     cluster_assignments = analyze_clusters(som, data, labels)
#
#     # Print cluster assignments
#     for node, texts in cluster_assignments.items():
#         print(f"Cluster at node {node}: {', '.join(texts)}")
#
#     # Final visualization of the best SOM and clusters
#     print("SOM based on cosine similarity:")
#     visualize_som(som, data, labels, title='SOM Visualization')
#     visualize_clusters(som, cluster_assignments, title='SOM Cluster Visualization')
#
#     # U-Matrix Visualization
#     plot_umatrix(som, title='SOM U-Matrix Visualization')
#     return context_cosine_similarity
# # Example usage:
# pdf_path1 = r'C:\Users\skykn\Downloads\Untitled document (15).pdf'
# pdf_path2 = r'C:\Users\skykn\Downloads\Untitled document (16).pdf'
#
# comparing_context(pdf_path1, pdf_path2)
