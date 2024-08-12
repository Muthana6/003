import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Plotting U-Matrix (Heatmap for Node Activation)
plt.figure(figsize=(8, 8))
plt.title('U-Matrix with Document Positions')
umatrix = som.distance_map().T
plt.pcolor(umatrix, cmap='bone_r')
plt.colorbar()

# Plot document positions
for i, (x, y) in enumerate(bmus):
    plt.text(x + 0.5, y + 0.5, f'Doc {i+1}', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

plt.show()

# Assigning themes manually for simplicity in this example
themes = ["Animals", "Noise", "Friendship"]
colors = ["red", "blue", "green"]

# Plotting with color coding
plt.figure(figsize=(8, 8))
plt.title('SOM with Color-Coded Themes')

for i, (x, y) in enumerate(bmus):
    plt.text(x + 0.5, y + 0.5, f'Doc {i+1}\n{themes[i]}', ha='center', va='center', color=colors[i], fontsize=12, weight='bold')

plt.pcolor(umatrix, cmap='bone_r')
plt.colorbar()
plt.show()

# Function to create word cloud for documents in each node
def create_word_cloud(documents, feature_names):
    text = " ".join(documents)
    wordcloud = WordCloud(width=300, height=300, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

# Plotting word clouds for each node
plt.figure(figsize=(10, 10))
plt.suptitle('Node Interpretation with Word Clouds')
for i in range(3):
    for j in range(3):
        plt.subplot(3, 3, i * 3 + j + 1)
        node_docs = [docs[k] for k in range(len(docs)) if bmus[k][0] == i and bmus[k][1] == j]
        if node_docs:
            create_word_cloud(node_docs, feature_names)
        plt.title(f'Node ({i}, {j})')

plt.tight_layout()
plt.show()
