import re

import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from nltk.corpus import stopwords


# Initializing variables for word2vec model.
num_features = 300
min_word_count = 40
num_workers = 8
context = 10
downsampling = 1e-3
word2vec_model_path = "./autograde_essay/models/word2vecmodel.bin"


def essay_to_wordlist(essay_v: str, remove_stopwords: bool) -> tuple:
    """Remove the tagged labels and word tokenize the sentence"""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


def essay_to_sentences(essay_v: str, remove_stopwords: bool) -> list:
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def make_feature_vec(
    words: list, model: Word2Vec | KeyedVectors, num_features: int
) -> np.array:
    """Make ar from the words list of an Essay."""
    feature_vec = np.zeros((num_features,), dtype="float32")
    num_words = 0
    if hasattr(model, "wv"):
        index2word_set = set(model.wv.index_to_key)
    else:
        index2word_set = set(model.index_to_key)

    for word in words:
        if word in index2word_set:
            num_words += 1
            if hasattr(model, "wv"):
                feature_vec = np.add(feature_vec, model.wv[word])
            else:
                feature_vec = np.add(feature_vec, model[word])
    feature_vec = np.divide(feature_vec, num_words)
    return feature_vec


def get_avg_feature_vecs(
    essays: list, model: Word2Vec | KeyedVectors, num_features: int
) -> np.array:
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essay_feature_vecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essay_feature_vecs[counter] = make_feature_vec(essay, model, num_features)
        counter += 1
    return essay_feature_vecs


def prep_train_data(train_data: pd.DataFrame) -> tuple:
    """Prepare data for training

    Args:
        data (pd.DataFrame): Input raw dataframe

    Returns:
        tuple: Tuple of features np.array and answers np.array
    """
    print("NLTK punkt downloading started")
    nltk.download("punkt")
    print("Download finished.\n")
    print("NLTK punkt downloading started")
    nltk.download("stopwords")
    print("Download finished.\n")

    train_data = train_data.dropna(axis=1)
    scores = train_data["domain1_score"]
    train_data = train_data["essay"]

    sentences = []

    for essay in train_data:
        sentences += essay_to_sentences(essay, remove_stopwords=True)

    print("Training Word2Vec Model...")
    model = Word2Vec(
        sentences,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling,
    )
    print("Word2Vec Model trained successfully!")
    model.init_sims(replace=True)
    print("Saving Word2Vec Model...")
    model.wv.save_word2vec_format(word2vec_model_path, binary=True)
    print("Word2Vec Model saved successfully!\n")

    clean_train_essays = []

    for essay_text in train_data:
        clean_train_essays.append(essay_to_wordlist(essay_text, remove_stopwords=True))
    train_vectors = get_avg_feature_vecs(clean_train_essays, model, num_features)

    return np.array(train_vectors), np.array(scores)


def prep_test_data(test_data: pd.DataFrame) -> np.array:
    """Prepare data for testing

    Args:
        data (pd.DataFrame): Input raw dataframe

    Returns:
        np.array: Features np.array
    """

    print("Loading Word2Vec Model ...")
    model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    print("Loading finished.")

    test_data = test_data.dropna(axis=1)
    test_data = test_data["essay"]

    sentences = []

    for essay in test_data:
        sentences += essay_to_sentences(essay, remove_stopwords=True)

    clean_test_essays = []
    for essay_text in test_data:
        clean_test_essays.append(essay_to_wordlist(essay_text, remove_stopwords=True))

    test_vectors = get_avg_feature_vecs(clean_test_essays, model, num_features)

    return np.array(test_vectors)
