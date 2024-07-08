from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf_from_freq_analysis(docs_freq_analysis):
    # Convert frequency analysis to strings for each document
    documents = []
    for doc in docs_freq_analysis:
        doc_string = ' '.join([word * freq for word, freq in doc])
        documents.append(doc_string)

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer and transform the documents into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Convert the TF-IDF matrix to an array for easier manipulation
    tfidf_matrix_array = tfidf_matrix.toarray()

    # Return TF-IDF matrix and feature names (terms)
    return tfidf_matrix_array, tfidf_vectorizer.get_feature_names_out()



