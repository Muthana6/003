# similarity_calculator.py

def calculate_combined_similarity(word_similarity, context_similarity, word_weight=0.5, context_weight=0.5):
    combined_similarity = (word_weight * word_similarity) + (context_weight * context_similarity)
    return combined_similarity
