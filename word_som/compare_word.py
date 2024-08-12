import fitz
from sklearn.metrics.pairwise import cosine_similarity
from Initial_Similarity_Assessment import compute_similarity
from word_extraction_and_normalization import extract_and_normalize_words
from stopword_removal import remove_stopwords
from stemming_or_limmatization import lemmatize_words
from text_analysis import calculate_term_frequency
from Vectorization import calculate_tfidf_from_freq_analysis
from word_som_train import create_and_train_som
from word_viz import visualize_som_simple


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def compare_words(pdf_paths, x_dim=5, y_dim=5):
    # List to store processed text from each document
    processed_docs = []
    document_labels = []

    for idx, pdf_path in enumerate(pdf_paths):
        # Extract and preprocess text from the PDF
        doc = extract_text_from_pdf(pdf_path)

        # Extract and normalize words
        doc_after_extraction = extract_and_normalize_words(doc)

        # Remove stopwords
        doc_after_removal = remove_stopwords(doc_after_extraction)

        # Lemmatize words
        doc_lemmatization = lemmatize_words(doc_after_removal)

        # Calculate term frequency
        text_analysis_doc = calculate_term_frequency(doc_lemmatization)

        processed_docs.append(text_analysis_doc)
        document_labels.append(f'Document {idx + 1}')

    # Calculate TF-IDF vectors
    tfidf_results, feature_names = calculate_tfidf_from_freq_analysis(processed_docs)

    # Calculate similarity using the defined function
    similarity_matrix = compute_similarity(tfidf_results)
    print('Similarity matrix:', similarity_matrix)

    # Create and train the SOM
    som = create_and_train_som(tfidf_results, x=x_dim, y=y_dim)

    # Visualize the SOM with document labels
    visualize_som_simple(som, tfidf_results, document_labels, title='Document Positions on the Self-Organizing Map')

    # Print the cosine similarity between documents
    cosine_word_similarity = cosine_similarity(tfidf_results)
    print('Cosine similarity between documents:')
    for i in range(len(pdf_paths)):
        for j in range(i + 1, len(pdf_paths)):
            print(f'Document {i + 1} and Document {j + 1}: {cosine_word_similarity[i][j]:.4f}')

    # Print the similarity list
    print("Document Similarities:")
    if isinstance(similarity_matrix, list):  # Check if the similarity matrix is a list
        if len(similarity_matrix) == 1:  # Handle the case for only two documents
            print(f'Document 1 and Document 2: {similarity_matrix[0]:.4f}')
        else:
            for i in range(len(similarity_matrix)):
                print(f'Document {i + 1} Similarity with other documents:')
                for j in range(i + 1, len(similarity_matrix)):
                    print(f'  with Document {j + 1}: {similarity_matrix[j]:.4f}')
    else:
        print(f'Document 1 and Document 2: {similarity_matrix:.4f}')

    print('Term frequency for all documents:')
    for i, doc in enumerate(processed_docs):
        print(f'Document {i + 1}:')
        print(doc)

    print("TF-IDF Matrix:")
    print(tfidf_results)
    print("Feature Names:")
    # print(feature_names)

    return tfidf_results


# Example usage:
pdf_paths = [r'../pdf data/pdf_1.pdf', r'../pdf data/pdf_2.pdf', r'../pdf data/pdf_3.pdf',r'../pdf data/pdf_4.pdf',r'../pdf data/pdf_5.pdf']
word_vectors = compare_words(pdf_paths)
print('TF-IDF vectors:', word_vectors)
