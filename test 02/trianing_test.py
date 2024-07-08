import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from minisom import MiniSom
def compute_similarity(tfidf_matrix):
    """
    Compute the cosine similarity between document vectors.

    Parameters:
    tfidf_matrix (sparse matrix): The TF-IDF matrix.

    Returns:
    similarity_dict: A dictionary where keys are document indices and values are dictionaries with similarity scores to other documents.
    """
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    similarity_dict = {i: {j: similarity_matrix[i][j] for j in range(len(similarity_matrix[i])) if i != j} for i in range(len(similarity_matrix))}
    return similarity_dict

def plot_similarity_heatmap(similarity_dict, labels):
    """
    Plot a heatmap for document similarity, excluding self-similarity.

    Parameters:
    similarity_dict (dict): A dictionary containing similarity scores between documents.
    labels (list): A list of document labels for the heatmap.
    """
    size = len(labels)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                similarity_matrix[i][j] = np.nan
            else:
                similarity_matrix[i][j] = similarity_dict.get(i, {}).get(j, 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=labels, yticklabels=labels, mask=np.isnan(similarity_matrix))
    plt.title('Document Similarity Heatmap (excluding self-similarity)')
    plt.xlabel('Documents')
    plt.ylabel('Documents')
    plt.show()

def create_and_train_som(tfidf_matrix, x_dim=5, y_dim=5):
    som = MiniSom(x_dim, y_dim, tfidf_matrix.shape[1], sigma=0.5, learning_rate=0.5)
    som.random_weights_init(tfidf_matrix.toarray())
    som.train_random(tfidf_matrix.toarray(), 100)
    return som

def plot_som(som, feature_names, x_dim, y_dim):
    plt.figure(figsize=(10, 10))
    for i, x in enumerate(tfidf_matrix.toarray()):
        w = som.winner(x)
        plt.text(w[0] + .5, w[1] + .5, str(i), fontsize=12, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.imshow(som.distance_map().T, cmap='bone_r', alpha=.5)
    plt.colorbar()
    plt.title('Self-Organizing Map (SOM)')
    plt.show()

documents = [
    "I love programming in Python",
    "Python is a great programming language",

]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)
similarity_dict = compute_similarity(tfidf_matrix)
document_labels = ['Document 1', 'Document 2']

plot_similarity_heatmap(similarity_dict, document_labels)

som = create_and_train_som(tfidf_matrix)
plot_som(som, vectorizer.get_feature_names_out(), 5, 5)
