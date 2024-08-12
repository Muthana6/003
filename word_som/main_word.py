

                # testing ,,,,,,,,,,,,,,,



# import fitz
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from Initial_Similarity_Assessment import compute_similarity
# from word_viz import extract_text_from_pdf
# from stemming_or_limmatization import lemmatize_words
# from stopword_removal import remove_stopwords
# from text_analysis import calculate_term_frequency
# from Vectorization import calculate_tfidf_from_freq_analysis
# from word_extraction_and_normalization import extract_and_normalize_words
# from word_som_train import create_and_train_som
# from word_viz2 import plot_som_documents
# import matplotlib.pyplot as plt
#
# def compare_word(pdf_path1, pdf_path2, x_dim=5, y_dim=5):
#     # Extract and preprocess text from the PDFs
#     doc1 = extract_text_from_pdf(pdf_path1)
#     doc2 = extract_text_from_pdf(pdf_path2)
#
#     # Extract and normalize words for both documents
#     doc1_after_extraction = extract_and_normalize_words(doc1)
#     doc2_after_extraction = extract_and_normalize_words(doc2)
#
#     # Remove stopwords for both documents
#     doc1_after_removal = remove_stopwords(doc1_after_extraction)
#     doc2_after_removal = remove_stopwords(doc2_after_extraction)
#
#     # Lemmatize words for both documents
#     doc1_lemmatization = lemmatize_words(doc1_after_removal)
#     doc2_lemmatization = lemmatize_words(doc2_after_removal)
#
#     # Calculate term frequency for both documents
#     text_analysis_doc1 = calculate_term_frequency(doc1_lemmatization)
#     text_analysis_doc2 = calculate_term_frequency(doc2_lemmatization)
#
#     docs_freq_analysis = [text_analysis_doc1, text_analysis_doc2]
#
#     # Calculate TF-IDF vectors
#     tfidf_results, feature_names = calculate_tfidf_from_freq_analysis(docs_freq_analysis)
#
#     # Calculate similarity using the defined function
#     similarity_list = compute_similarity(tfidf_results)
#
#     # Create and train the SOM
#     som = create_and_train_som(tfidf_results, x=x_dim, y=y_dim)
#
#     # Define document labels for visualization purposes
#     document_labels = ['Document 1', 'Document 2']
#
#     # Visualize the SOM with document labels
#     plot_som_documents(som, tfidf_results, document_labels)
#
#     # Print the cosine similarity between documents
#     cosine_word_similarity = cosine_similarity(tfidf_results)
#     print('Cosine similarity between Document 1 and Document 2:', cosine_word_similarity[0][1])
#
#     # Print the similarity list
#     print("Document 1 Similarity with other documents:")
#     for idx, similarity in enumerate(similarity_list):
#         print(f"  with Document {idx + 1}: {similarity:.4f}")
#
#     print('Extract and normalize words')
#     print(doc1_after_extraction)
#     print(doc2_after_extraction)
#
#     print('Remove stopwords')
#     print(doc1_after_removal)
#     print(doc2_after_removal)
#
#     print('Lemmatize words')
#     print(doc1_lemmatization)
#     print(doc2_lemmatization)
#
#     print('Term frequency for both documents')
#     print(text_analysis_doc1)
#     print(text_analysis_doc2)
#
#     print("TF-IDF Matrix:")
#     print(tfidf_results)
#     print("Feature Names:")
#     print(feature_names)
#     return cosine_word_similarity[0][1]
#
# # Example usage:
# pdf_path1 = r'C:\Users\skykn\Downloads\Untitled document (15).pdf'
# pdf_path2 = r'C:\Users\skykn\Downloads\Untitled document (16).pdf'
#
# compare_word(pdf_path1, pdf_path2)
