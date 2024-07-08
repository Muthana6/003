import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')


def semantic_syntactic_analysis(text):
    """
    Perform semantic and syntactic analysis on the text.

    Parameters:
    text (str): The text to analyze.

    Returns:
    dict: A dictionary containing POS tags, dependency parse, and word co-occurrence information.
    """
    # Create a spaCy document
    doc = nlp(text)

    # Extract POS tags and dependencies
    pos_tags = [(token.text, token.pos_) for token in doc]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    # Analyze Word Co-occurrence within the same sentence
    word_cooccurrence = {}
    for token in doc:
        if token.text not in word_cooccurrence:
            word_cooccurrence[token.text] = {}
        for neighbor in token.children:
            if neighbor.text not in word_cooccurrence[token.text]:
                word_cooccurrence[token.text][neighbor.text] = 1
            else:
                word_cooccurrence[token.text][neighbor.text] += 1

    return {
        "pos_tags": pos_tags,
        "dependencies": dependencies,
        "word_cooccurrence": word_cooccurrence
    }
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text