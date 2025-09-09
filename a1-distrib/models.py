# models.py

from sentiment_data import *
from utils import *

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
        """
        Extract unigram features from a sentence.
        """
        features = Counter()
        for word in sentence:
            # Convert to lowercase for consistency
            word = word.lower()
            
            # Skip very short words (likely punctuation or noise)
            if len(word) < 2:
                continue
                
            # Get or add index for this word
            if add_to_indexer:
                word_idx = self.indexer.add_and_get_index(word, add=True)
            else:
                word_idx = self.indexer.add_and_get_index(word, add=False)
            
            # Only add feature if word is in indexer (seen during training)
            if word_idx != -1:
                features[word_idx] += 1
        
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
        """
        Extract bigram features from a sentence.
        """
        features = Counter()
        
        # Add unigram features as well
        for word in sentence:
            word = word.lower()
            # Skip very short words
            if len(word) < 2:
                continue
                
            if add_to_indexer:
                word_idx = self.indexer.add_and_get_index(word, add=True)
            else:
                word_idx = self.indexer.add_and_get_index(word, add=False)
            
            if word_idx != -1:
                features[word_idx] += 1
        
        # Add bigram features
        for i in range(len(sentence) - 1):
            word1 = sentence[i].lower()
            word2 = sentence[i+1].lower()
            
            # Skip if either word is too short
            if len(word1) < 2 or len(word2) < 2:
                continue
                
            bigram_str = f"{word1}_{word2}"
            
            if add_to_indexer:
                bigram_idx = self.indexer.add_and_get_index(bigram_str, add=True)
            else:
                bigram_idx = self.indexer.add_and_get_index(bigram_str, add=False)
            
            if bigram_idx != -1:
                features[bigram_idx] += 1
        
        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract advanced features including unigrams, bigrams, and other linguistic features.
        """
        features = Counter()
        
        # Unigram features with better preprocessing
        for word in sentence:
            word = word.lower()
            # Skip very short words
            if len(word) < 2:
                continue
                
            if add_to_indexer:
                word_idx = self.indexer.add_and_get_index(word, add=True)
            else:
                word_idx = self.indexer.add_and_get_index(word, add=False)
            
            if word_idx != -1:
                features[word_idx] += 1
        
        # Bigram features
        for i in range(len(sentence) - 1):
            word1 = sentence[i].lower()
            word2 = sentence[i+1].lower()
            
            # Skip if either word is too short
            if len(word1) < 2 or len(word2) < 2:
                continue
                
            bigram = f"{word1}_{word2}"
            if add_to_indexer:
                bigram_idx = self.indexer.add_and_get_index(bigram, add=True)
            else:
                bigram_idx = self.indexer.add_and_get_index(bigram, add=False)
            
            if bigram_idx != -1:
                features[bigram_idx] += 1
        
        # Position-based features
        if len(sentence) > 0:
            first_word = sentence[0].lower()
            if len(first_word) >= 2:
                first_feature = f"first_word_{first_word}"
                if add_to_indexer:
                    first_idx = self.indexer.add_and_get_index(first_feature, add=True)
                else:
                    first_idx = self.indexer.add_and_get_index(first_feature, add=False)
                
                if first_idx != -1:
                    features[first_idx] = 1
        
        if len(sentence) > 1:
            last_word = sentence[-1].lower()
            if len(last_word) >= 2:
                last_feature = f"last_word_{last_word}"
                if add_to_indexer:
                    last_idx = self.indexer.add_and_get_index(last_feature, add=True)
                else:
                    last_idx = self.indexer.add_and_get_index(last_feature, add=False)
                
                if last_idx != -1:
                    features[last_idx] = 1
        
        # Length-based features (binned)
        length = len(sentence)
        if length <= 5:
            length_bin = "short"
        elif length <= 15:
            length_bin = "medium"
        else:
            length_bin = "long"
            
        length_feature = f"length_{length_bin}"
        if add_to_indexer:
            length_idx = self.indexer.add_and_get_index(length_feature, add=True)
        else:
            length_idx = self.indexer.add_and_get_index(length_feature, add=False)
        
        if length_idx != -1:
            features[length_idx] = 1
        
        # Enhanced sentiment word features
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'outstanding', 'superb', 'marvelous', 'love', 'best', 'perfect', 'awesome', 'incredible']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'boring', 'stupid', 'ridiculous', 'pathetic', 'worst', 'hate', 'hated', 'disgusting', 'annoying', 'stupid']
        
        pos_count = sum(1 for word in sentence if word.lower() in positive_words)
        neg_count = sum(1 for word in sentence if word.lower() in negative_words)
        
        if pos_count > 0:
            pos_feature = f"has_positive_words"
            if add_to_indexer:
                pos_idx = self.indexer.add_and_get_index(pos_feature, add=True)
            else:
                pos_idx = self.indexer.add_and_get_index(pos_feature, add=False)
            
            if pos_idx != -1:
                features[pos_idx] = 1
        
        if neg_count > 0:
            neg_feature = f"has_negative_words"
            if add_to_indexer:
                neg_idx = self.indexer.add_and_get_index(neg_feature, add=True)
            else:
                neg_idx = self.indexer.add_and_get_index(neg_feature, add=False)
            
            if neg_idx != -1:
                features[neg_idx] = 1
        
        # Punctuation features
        exclamation_count = sum(1 for word in sentence if '!' in word)
        question_count = sum(1 for word in sentence if '?' in word)
        
        if exclamation_count > 0:
            excl_feature = f"has_exclamation"
            if add_to_indexer:
                excl_idx = self.indexer.add_and_get_index(excl_feature, add=True)
            else:
                excl_idx = self.indexer.add_and_get_index(excl_feature, add=False)
            
            if excl_idx != -1:
                features[excl_idx] = 1
        
        if question_count > 0:
            quest_feature = f"has_question"
            if add_to_indexer:
                quest_idx = self.indexer.add_and_get_index(quest_feature, add=True)
            else:
                quest_idx = self.indexer.add_and_get_index(quest_feature, add=False)
            
            if quest_idx != -1:
                features[quest_idx] = 1
        
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
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        """
        Predict sentiment using perceptron weights.
        """
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        
        # Compute dot product of weights and features
        score = 0.0
        for feature_idx, count in features.items():
            if feature_idx < len(self.weights):
                score += self.weights[feature_idx] * count
        
        # Return 1 if score >= 0, 0 otherwise
        return 1 if score >= 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor
    
    def predict(self, sentence: List[str]) -> int:
        """
        Predict sentiment using logistic regression weights.
        """
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        
        # Compute dot product of weights and features
        score = 0.0
        for feature_idx, count in features.items():
            if feature_idx < len(self.weights):
                score += self.weights[feature_idx] * count
        
        # Apply sigmoid function with numerical stability
        import math
        if score > 500:
            probability = 1.0
        elif score < -500:
            probability = 0.0
        else:
            probability = 1.0 / (1.0 + math.exp(-score))
        
        # Return 1 if probability >= 0.5, 0 otherwise
        return 1 if probability >= 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    import random
    import numpy as np
    
    # First pass: build vocabulary and feature space
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    # Initialize weights to zero
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features, dtype=np.float64)
    
    # Training loop with random shuffling and learning rate scheduling
    num_epochs = 20
    initial_learning_rate = 1.0
    
    for epoch in range(num_epochs):
        # Random shuffle the training data each epoch
        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)
        
        # Learning rate scheduling: decrease over time
        learning_rate = initial_learning_rate / (1.0 + epoch * 0.1)
        
        for ex in shuffled_exs:
            # Extract features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            
            # Compute prediction
            score = 0.0
            for feature_idx, count in features.items():
                if feature_idx < len(weights):
                    score += weights[feature_idx] * count
            
            prediction = 1 if score >= 0 else 0
            
            # Update weights if prediction is wrong
            if prediction != ex.label:
                update = learning_rate * (ex.label - prediction)
                for feature_idx, count in features.items():
                    if feature_idx < len(weights):
                        weights[feature_idx] += update * count
    
    return PerceptronClassifier(weights.tolist(), feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    import math
    import random
    import numpy as np
    
    # First pass: build vocabulary and feature space
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    
    # Initialize weights to zero
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features, dtype=np.float64)
    momentum = np.zeros(num_features, dtype=np.float64)
    
    # Training loop with gradient descent
    num_epochs = 800
    initial_learning_rate = 10.0
    regularization = 0.0
    momentum_factor = 0.98
    
    for epoch in range(num_epochs):
        # Random shuffle the training data each epoch
        shuffled_exs = train_exs.copy()
        random.shuffle(shuffled_exs)
        
        # Learning rate scheduling - step decay
        if epoch < 200:
            learning_rate = initial_learning_rate
        elif epoch < 400:
            learning_rate = initial_learning_rate * 0.5
        elif epoch < 600:
            learning_rate = initial_learning_rate * 0.25
        else:
            learning_rate = initial_learning_rate * 0.125
        
        # Compute gradients
        gradients = np.zeros(num_features, dtype=np.float64)
        
        for ex in shuffled_exs:
            # Extract features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            
            # Compute prediction probability
            score = 0.0
            for feature_idx, count in features.items():
                if feature_idx < len(weights):
                    score += weights[feature_idx] * count
            
            # Apply sigmoid with numerical stability
            if score > 500:
                probability = 1.0
            elif score < -500:
                probability = 0.0
            else:
                probability = 1.0 / (1.0 + math.exp(-score))
            
            # Compute gradient
            error = ex.label - probability
            
            for feature_idx, count in features.items():
                if feature_idx < len(gradients):
                    gradients[feature_idx] += error * count
        
        # Update weights with momentum and regularization
        for i in range(len(weights)):
            # Compute momentum update
            momentum[i] = momentum_factor * momentum[i] + learning_rate * (gradients[i] - regularization * weights[i])
            # Update weights
            weights[i] += momentum[i]
    
    return LogisticRegressionClassifier(weights.tolist(), feat_extractor)


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
