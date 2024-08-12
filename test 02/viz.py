import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from minisom import MiniSom
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Example documents
docs = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barks loudly.",
    "The quick fox and the dog are friends."
]

# Preprocess and convert to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(docs).toarray()
feature_names = vectorizer.get_feature_names_out()

# Create and train the SOM
som = MiniSom(x=3, y=3, input_len=tfidf_matrix.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(tfidf_matrix)
som.train_random(tfidf_matrix, num_iteration=100)

# Find BMUs for each document
bmus = np.array([som.winner(x) for x in tfidf_matrix])

# Get SOM dimensions
som_x, som_y = som.get_weights().shape[:2]

# Component Planes Visualization
plt.figure(figsize=(10, 10))
for i, feature in enumerate(feature_names):
    plt.subplot(5, 5, i + 1)
    plt.title(feature)
    plt.pcolor(som.get_weights()[:, :, i].T, cmap='coolwarm')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Label Maps Visualization
labels = ["Animals", "Noise", "Friendship"]
plt.figure(figsize=(8, 8))
for i, (x, y) in enumerate(bmus):
    plt.text(x + 0.5, y + 0.5, f'Doc {i+1}\n{labels[i]}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
plt.xticks(np.arange(som_x))
plt.yticks(np.arange(som_y))
plt.grid()
plt.title('SOM with Color-Coded Themes')
plt.show()
