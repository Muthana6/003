import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from minisom import MiniSom
import matplotlib.pyplot as plt
from word_som.stemming_or_limmatization import lemmatize_words
from word_som.stopword_removal import remove_stopwords
from word_som.text_analysis import calculate_term_frequency
from word_som.Vectorization import calculate_tfidf_from_freq_analysis
from word_som.word_extraction_and_normalization import extract_and_normalize_words

# Step 1: Scrape the website to get PDF links
def get_arxiv_pdf_links(search_query, max_results=5):
    base_url = "https://export.arxiv.org/api/query?search_query="
    query = f"{search_query}&start=0&max_results={max_results}"
    response = requests.get(base_url + query)
    soup = BeautifulSoup(response.content, 'xml')
    links = [entry.find('id').text.replace('abs', 'pdf') + '.pdf' for entry in soup.find_all('entry')]
    return links

# Step 2: Download PDFs
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Step 3: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Step 4: Preprocess the text
def preprocess_text(text):
    words = extract_and_normalize_words(text)
    words_no_stop = remove_stopwords(words)
    lemmatized_words = lemmatize_words(words_no_stop)
    return lemmatized_words  # Return a list of strings

# Step 5: Create and train SOM
def create_and_train_som(tfidf_matrix, x=5, y=5, input_len=None, sigma=1.0, learning_rate=0.5):
    if input_len is None:
        input_len = tfidf_matrix.shape[1]
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(tfidf_matrix)
    som.train_random(tfidf_matrix, num_iteration=100)
    return som

# Step 6: Visualize the SOM
def plot_som(som, feature_names, x_dim, y_dim):
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(som.get_weights()):
        for j, y in enumerate(x):
            plt.text(i, j, feature_names[np.argmax(y)], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.xticks(np.arange(x_dim))
    plt.yticks(np.arange(y_dim))
    plt.grid()
    plt.title('Self-Organizing Map of TF-IDF Features')
    plt.show()

def plot_document_positions(som, data, labels):
    plt.figure(figsize=(10, 10))
    for i, doc in enumerate(data):
        w = som.winner(doc)
        plt.text(w[0], w[1], labels[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.xticks(np.arange(som.get_weights().shape[0]))
    plt.yticks(np.arange(som.get_weights().shape[1]))
    plt.grid()
    plt.title('Document Positions on the Self-Organizing Map')
    plt.show()

# Main Script
pdf_path_local = r'C:\Users\skykn\OneDrive\Desktop\BA Kutiba\pdf_1.pdf'
search_query = "machine learning"  # Adjust the search query as needed
pdf_links = get_arxiv_pdf_links(search_query, max_results=5)

pdf_paths = []
texts = []

# Add local PDF text extraction
text_local = extract_text_from_pdf(pdf_path_local)
if text_local:
    texts.append(text_local)
    pdf_paths.append(pdf_path_local)
else:
    print(f"Local PDF not found at {pdf_path_local}")

# Download and extract text from PDFs
for i, link in enumerate(pdf_links):
    pdf_path = f'pdf_{i+1}.pdf'
    download_pdf(link, pdf_path)
    pdf_paths.append(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    if text:
        texts.append(text)

# Preprocess texts
processed_texts = [' '.join(preprocess_text(text)) for text in texts]

# Calculate TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_texts)
feature_names = vectorizer.get_feature_names_out()

# Compute similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Train SOM
som = create_and_train_som(tfidf_matrix.toarray())

# Visualize SOM
plot_som(som, feature_names, x_dim=5, y_dim=5)

# Visualize document positions on SOM
document_labels = ['Local Document'] + [f'Document {i+1}' for i in range(len(pdf_links))]
plot_document_positions(som, tfidf_matrix.toarray(), document_labels)

# Print the cosine similarity between documents
print("First Row of Document Similarity Matrix:")
print(similarity_matrix[0])
