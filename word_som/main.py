#
#
#
# # Empty for testing
#
#
#
#
#
#
#
#
#
# # main.py
# from word_extraction_and_normalization import extract_and_normalize_words
# from stopword_removal import remove_stopwords
# from stemming_or_limmatization import lemmatize_words
# from text_analysis import calculate_term_frequency
# from Vectorization import calculate_tfidf_from_freq_analysis
# from Initial_Similarity_Assessment import compute_similarity
# from word_som_train import create_and_train_som
# from word_viz import plot_som
# from word_viz2 import plot_similarity_heatmap
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the dimensions of the SOM
# x_dim = 5
# y_dim = 5
#
# # Example documents
# doc1 = ("When we study the many aspects of computing and computers, it is important to know about the history of computers. "
#     "Charles Babbage designed an Analytical Engine which was a general computer. It helps us understand the growth and progress of technology through the times. "
#     "It is also an important topic for competitive and banking exams.")
# doc2 = "In the heart of ancient Mesopotamia, the cradle of civilization blossomed with the first known cities. The pyramids of Egypt stood as monumental testaments to the ingenuity and power of the pharaohs. Across the Aegean Sea, the city-states of Greece birthed democracy and philosophical thought. Meanwhile, Rome's vast empire spread its influence, laying the foundations for modern law and governance."
# doc3 = "Machine learning and data science are fascinating fields that are evolving rapidly. They offer great potential for innovation and solving complex problems."
#
# # Combine all documents for processing
# documents = [doc1, doc2, doc3]
#
# # Process each document
# processed_docs = []
# for doc in documents:
#     words = extract_and_normalize_words(doc)
#     words_no_stop = remove_stopwords(words)
#     lemmatized_words = lemmatize_words(words_no_stop)
#     term_freqs = calculate_term_frequency(lemmatized_words)
#     processed_docs.append(term_freqs)
#
# # Vectorization and SOM training
# tfidf_results, feature_names = calculate_tfidf_from_freq_analysis(processed_docs)
# similarity_dict = compute_similarity(tfidf_results)
# som = create_and_train_som(tfidf_results, x=x_dim, y=y_dim)
#
# # Visualize the SOM
# plot_som(som, feature_names, x_dim, y_dim)
#
# # Define document labels for visualization purposes (can be titles if available)
# document_labels = ['Document 1', 'Document 2', 'Document 3']
#
# # Visualize document similarity
# plot_similarity_heatmap(similarity_dict, document_labels)
#
# # Print the similarity dictionary
# print("Document Similarity:")
# for doc_index, similarities in similarity_dict.items():
#     print(f"Document {doc_index} similarities:")
#     for other_doc, score in similarities.items():
#         print(f"  with Document {other_doc}: {score:.4f}")
#
# # Print the processing steps
# print('Extract and normalize words')
# print(processed_docs[0])
# print(processed_docs[1])
# print(processed_docs[2])
#
# print('Remove stopwords')
# print(processed_docs[0])
# print(processed_docs[1])
# print(processed_docs[2])
#
# print('Lemmatize words')
# print(processed_docs[0])
# print(processed_docs[1])
# print(processed_docs[2])
#
# print('Term frequency for both documents')
# print(processed_docs[0])
# print(processed_docs[1])
# print(processed_docs[2])
#
# print("TF-IDF Matrix:")
# print(tfidf_results)
# print("Feature Names:")
# print(feature_names)
#
