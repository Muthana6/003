import re  # Import the regular expression module


def extract_and_normalize_words(document):
    """
    Extracts and normalizes words from a given text document.

    Parameters:
    document (str): The text document from which words are to be extracted.

    Returns:
    list: A list of extracted and normalized words.
    """
    # Normalize the document to lowercase to ensure uniformity
    document = document.lower()

    # Use regular expression to find all alphabetic words in the document.
    # \b[a-z]+\b is a regex pattern that matches whole words ('a-z' matches any lowercase letter, '+' ensures one or more letters, and '\b' asserts position at a word boundary)
    words = re.findall(r'\b[a-z]+\b', document)

    return words