import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """

    # Step 1: Tokenize documents
    tokenized_docs = [doc.split() for doc in documents]

    # Step 2: Build vocabulary (sorted for consistent order)
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    N = len(documents)

    # Step 3: Compute Document Frequency
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1

    # Step 4: Compute IDF
    idf = {word: math.log(N / df[word]) for word in vocab}

    # Step 5: Create TF-IDF matrix
    tfidf_matrix = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized_docs):
        word_count = Counter(doc)
        total_terms = len(doc)

        for word, count in word_count.items():
            j = vocab_index[word]
            tf = count / total_terms
            tfidf_matrix[i][j] = tf * idf[word]

    return tfidf_matrix, vocab