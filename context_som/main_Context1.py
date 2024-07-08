



# Testing..........



# from preprocessing_tokenization import preprocess_text
# from semantic_syntacic import semantic_syntactic_analysis
# from Vector_Representation import get_word_vectors, aggregate_vectors
# from similarity_measurement import calculate_cosine_similarity
# from train_som import create_and_train_som
# import numpy as np
# import matplotlib.pyplot as plt
# from Analizing_context import analyze_clusters, visualize_som, visualize_clusters
#
#
#
# # Example texts to preprocess
# text1 = ("When we study the many aspects of computing and computers, it is important to know about the history of computers. "
#     "Charles Babbage designed an Analytical Engine which was a general computer. It helps us understand the growth and progress of technology through the times. "
#     "It is also an important topic for competitive and banking exams.")
# text2 = (
#     "In the heart of ancient Mesopotamia, the cradle of civilization blossomed with the first known cities. The pyramids of Egypt stood as monumental testaments to the ingenuity and power of the pharaohs. Across the Aegean Sea, the city-states of Greece birthed democracy and philosophical thought. Meanwhile, Rome's vast empire spread its influence, laying the foundations for modern law and governance.")
#
# # Preprocessing both texts
# words1, filtered_sentences1 = preprocess_text(text1)
# words2, filtered_sentences2 = preprocess_text(text2)
#
# # Print preprocessing results
# print("Preprocessed Words for Text 1:", words1)
# print("Preprocessed Sentences for Text 1:", filtered_sentences1)
# print("Preprocessed Words for Text 2:", words2)
# print("Preprocessed Sentences for Text 2:", filtered_sentences2)
#
# # Optional: Perform semantic and syntactic analysis (not used for vector representation in this context)
# analysis1 = semantic_syntactic_analysis(text1)
# analysis2 = semantic_syntactic_analysis(text2)
#
# # Print analysis results for verification
# print("Analysis Results for Text 1:", analysis1)
# print("Analysis Results for Text 2:", analysis2)
#
# # Parameters for iteration
# iterations = 3
# best_som = None
# best_cluster_assignments = None
# best_cosine_similarity = 0
#
# # for i in range(iterations):
# # print(f"Iteration {i + 1}/{iterations}")
#
# # Vector representation for both documents
# word_vectors1 = get_word_vectors(words1)
# word_vectors2 = get_word_vectors(words2)
# document_vector1 = aggregate_vectors(word_vectors1)
# document_vector2 = aggregate_vectors(word_vectors2)
#
# # Print document vectors to verify
# print("Document Vector for Text 1:", document_vector1)
# print("Document Vector for Text 2:", document_vector2)
#
# # Calculate cosine similarity between the two document vectors
# contex_cosine_similarity = calculate_cosine_similarity(document_vector1, document_vector2)
# print("Cosine Similarity between Text 1 and Text 2:", contex_cosine_similarity)
#
# # Document vectors array for further processing (e.g., SOM training)
# data = np.array([document_vector1, document_vector2])
# labels = ["Text 1", "Text 2"]
#
# # Create and train the SOM with a fixed random seed for consistency
# som = create_and_train_som(data, x_size=5, y_size=5, input_len=data.shape[1], sigma=0.3, learning_rate=0.5,
#                            num_iteration=5000, random_seed=42)
#
# # Print SOM results
# print("SOM Training Completed")
#
# # Retrieve and print the SOM weights
# som_weights = som.get_weights()
# print("SOM Weights:\n", som_weights)
#
# # Retrieve and print the winning nodes for the input vectors
# winning_nodes = [som.winner(vector) for vector in data]
# print("Winning Nodes for Input Vectors:", winning_nodes)
#
# # Visualize the SOM
# # visualize_som(som, data, labels, title=f'SOM Visualization - Iteration {i + 1}')
#
# # Analyze clusters
# cluster_assignments = analyze_clusters(som, data, labels)
#
# # Print cluster assignments
# for node, texts in cluster_assignments.items():
#     print(f"Cluster at node {node}: {', '.join(texts)}")
#
# # Visualize the clusters on the SOM grid
# # visualize_clusters(som, cluster_assignments, title=f'SOM Cluster Visualization - Iteration {i + 1}')
#
# # Save the best SOM based on cosine similarity
# if contex_cosine_similarity > best_cosine_similarity:
#     best_som = som
#     best_cluster_assignments = cluster_assignments
#     best_cosine_similarity = contex_cosine_similarity
#
# # Final visualization of the best SOM and clusters
# print("Best SOM based on cosine similarity:")
# visualize_som(best_som, data, labels, title='Best SOM Visualization')
# visualize_clusters(best_som, best_cluster_assignments, title='Best SOM Cluster Visualization')
#
#
# # if __name__ == "__main__":
# #     main()
#
#
#
#
#
#
#
