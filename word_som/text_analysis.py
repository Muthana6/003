from nltk.probability import FreqDist

# Input data
# words = ['hello', 'world', 'test', 'document', 'number', 'punctuation', 'explores', 'vast', 'expanse', 'language', 'weave', 'word', 'intricate', 'tapestry', 'meaning', 'across', 'page', 'story', 'unfold', 'like', 'petal', 'unfurl', 'morning', 'sun', 'sentence', 'brushstroke', 'painting', 'vivid', 'scene', 'upon', 'canvas', 'imagination', 'realm', 'prose', 'possibility', 'endless', 'stretch', 'beyond', 'horizon', 'comprehension', 'let', 'u', 'embark', 'journey', 'together', 'navigate', 'boundless', 'sea', 'expression']

# Function to calculate term frequency
def calculate_term_frequency(words):
    """
    Calculate the frequency of each word in a list of words.

    Parameters:
    words (list): A list of words from which to calculate frequencies.

    Returns:
    list: A list of tuples where each tuple contains a word and its frequency.
    """
    # Create a frequency distribution of the words
    freq_distribution = FreqDist(words)
    # Convert frequency distribution to list of tuples
    freq_list = [(word, freq_distribution[word]) for word in freq_distribution]
    return freq_list






