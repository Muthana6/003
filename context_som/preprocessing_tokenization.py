import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    """
    Process the input text: normalize, tokenize, and remove stopwords from both words and sentences.

    Parameters:
    text (str): The text to preprocess.

    Returns:
    tuple: A tuple containing a list of words without stopwords and a list of sentences without stopwords.
    """
    # Normalize text: convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization: split into words or sentences
    words = word_tokenize(text)
    original_sentences = sent_tokenize(text)

    # Stopwords removal for words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Stopwords removal for sentences
    filtered_sentences = []
    for sentence in original_sentences:
        # Tokenize the sentence into words
        sentence_words = word_tokenize(sentence)
        # Filter out the stopwords
        filtered_sentence_words = [word for word in sentence_words if word not in stop_words]
        # Reconstruct the sentence
        filtered_sentence = ' '.join(filtered_sentence_words)
        filtered_sentences.append(filtered_sentence)

    return filtered_words, filtered_sentences
