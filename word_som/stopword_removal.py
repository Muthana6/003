import nltk
from nltk.corpus import stopwords

# Function to ensure that necessary NLTK resources are downloaded only if they aren't already available
def download_nltk_resources():
    try:
        # Check if the stopwords are available, else download them
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # Download the stopwords if not present
        nltk.download('stopwords')
        print("Downloaded the stopwords dataset")


# Ensure that the stopwords dataset is available
download_nltk_resources()
def remove_stopwords(word_list):
    """
    Removes common stopwords from a list of words.

    Parameters:
    word_list (list): The list of words from which stopwords are to be removed.

    Returns:
    list: A list of words with stopwords removed.
    """
    # Retrieve the set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter the list to remove words that are in the stop words set
    filtered_words = [word for word in word_list if word not in stop_words]

    return filtered_words



