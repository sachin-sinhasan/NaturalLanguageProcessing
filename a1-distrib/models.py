# models.py

from sentiment_data import *
from utils import *
import math
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(word)
                features[idx] += 1
            else:
                idx = self.indexer.get_index(word)
                if idx != -1:
                    features[idx] += 1
        return features

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        # Create bigrams from the sentence
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(bigram)
                features[idx] += 1
            else:
                idx = self.indexer.get_index(bigram)
                if idx != -1:
                    features[idx] += 1
        return features
    

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    This example combines unigrams, bigrams, and adds a feature for sentence length.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        # Unigram features
        for word in sentence:
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(("UNI", word))
                features[idx] += 1
            else:
                idx = self.indexer.get_index(("UNI", word))
                if idx != -1:
                    features[idx] += 1
        # Bigram features
        for i in range(len(sentence) - 1):
            bigram = (sentence[i], sentence[i + 1])
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(("BI", bigram))
                features[idx] += 1
            else:
                idx = self.indexer.get_index(("BI", bigram))
                if idx != -1:
                    features[idx] += 1
        # Sentence length feature (bucketed)
        length_bucket = min(len(sentence) // 5, 5)
        if add_to_indexer:
            idx = self.indexer.add_and_get_index(("LEN", length_bucket))
            features[idx] = 1
        else:
            idx = self.indexer.get_index(("LEN", length_bucket))
            if idx != -1:
                features[idx] = 1
        return features



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Perceptron classifier for sentiment analysis.
    Stores the weight vector and feature extractor.
    """
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights  # Counter[int]: feature index -> weight
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for idx, value in features.items():
            score += self.weights.get(idx, 0.0) * value
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Logistic Regression classifier for sentiment analysis.
    Stores the weight vector and feature extractor.
    """
    def __init__(self, weights: Counter, feat_extractor: FeatureExtractor):
        self.weights = weights  # Counter[int]: feature index -> weight
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = 0.0
        for idx, value in features.items():
            score += self.weights.get(idx, 0.0) * value
        prob = 1.0 / (1.0 + math.exp(-score))
        return 1 if prob >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int = 5) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param num_epochs: number of passes over the training data
    :return: trained PerceptronClassifier model
    """
    weights = Counter()
    # Build feature indexer on training data
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    for epoch in range(num_epochs):
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights.get(idx, 0.0) * value for idx, value in features.items())
            pred = 1 if score >= 0 else 0
            gold = ex.label
            if pred != gold:
                # Update weights
                for idx, value in features.items():
                    weights[idx] += value * (1 if gold == 1 else -1)
    return PerceptronClassifier(weights, feat_extractor)



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int = 10, lr: float = 0.1) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :param num_epochs: number of passes over the training data
    :param lr: learning rate
    :return: trained LogisticRegressionClassifier model
    """
    weights = Counter()
    # Build feature indexer on training data
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    for epoch in range(num_epochs):
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights.get(idx, 0.0) * value for idx, value in features.items())
            prob = 1.0 / (1.0 + math.exp(-score))
            gold = ex.label
            # Gradient update
            for idx, value in features.items():
                weights[idx] += lr * (gold - prob) * value
    return LogisticRegressionClassifier(weights, feat_extractor)


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
