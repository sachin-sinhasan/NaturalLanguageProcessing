# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

"""
class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        # Average embeddings for words in ex_words
        vectors = [self.word_embeddings.get_embedding(word) for word in ex_words if self.word_embeddings.contains(word)]
        if not vectors:
            avg_vec = np.zeros(self.word_embeddings.get_embedding_dim())
        else:
            avg_vec = np.mean(vectors, axis=0)
        # Convert to torch tensor
        inp = torch.tensor(avg_vec, dtype=torch.float32).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inp)
            pred = torch.argmax(logits, dim=1).item()
        return pred


class DANNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.ff(x)

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings, train_model_for_typo_setting):
    input_dim = word_embeddings.get_embedding_dim()
    model = DANNet(input_dim, hidden_dim=128, dropout=0.2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Prepare training data
    def get_avg_embedding(ex):
        vectors = [word_embeddings.get_embedding(w) for w in ex.words if word_embeddings.contains(w)]
        if not vectors:
            return np.zeros(input_dim)
        return np.mean(vectors, axis=0)

    X_train = np.array([get_avg_embedding(ex) for ex in train_exs])
    y_train = np.array([ex.label for ex in train_exs])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return NeuralSentimentClassifier(model, word_embeddings)
"""

class DANNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim=128, dropout=0.2, padding_idx=0, embeddings_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if embeddings_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings_matrix))
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        # x: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        mask = (x != 0).unsqueeze(-1)  # (batch_size, seq_len, 1)
        emb = emb * mask  # zero out PAD embeddings
        summed = emb.sum(dim=1)  # (batch_size, embedding_dim)
        lengths = mask.sum(dim=1)  # (batch_size, 1)
        avg_emb = summed / lengths.clamp(min=1)  # avoid division by zero
        return self.ff(avg_emb)

def pad_batch(batch, pad_idx=0, max_len=None):
    # batch: list of list of word indices
    if max_len is None:
        max_len = max(len(x) for x in batch)
    padded = [x + [pad_idx] * (max_len - len(x)) if len(x) < max_len else x[:max_len] for x in batch]
    return torch.tensor(padded, dtype=torch.long)

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings, train_model_for_typo_setting):
    vocab_size = word_embeddings.vocab_size()
    embedding_dim = word_embeddings.get_embedding_dim()
    embeddings_matrix = word_embeddings.get_numpy_matrix()  # shape: (vocab_size, embedding_dim)
    model = DANNet(vocab_size, embedding_dim, hidden_dim=128, dropout=0.2, padding_idx=0, embeddings_matrix=embeddings_matrix)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Convert examples to indices
    def to_indices(ex):
        return [word_embeddings.word_to_index(w) for w in ex.words if word_embeddings.contains(w)]

    batch_size = 32
    train_indices = [to_indices(ex) for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]

    model.train()
    for epoch in range(args.epochs):
        # Shuffle for each epoch
        combined = list(zip(train_indices, train_labels))
        random.shuffle(combined)
        train_indices[:], train_labels[:] = zip(*combined)
        for i in range(0, len(train_indices), batch_size):
            batch = train_indices[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            batch_padded = pad_batch(batch, pad_idx=0)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(batch_padded)
            loss = loss_fn(logits, labels_tensor)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return NeuralSentimentClassifier(model, word_embeddings)

# Optionally, you can batch in predict_all as well:
class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model, word_embeddings):
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = [self.word_embeddings.word_to_index(w) for w in ex_words if self.word_embeddings.contains(w)]
        if not indices:
            indices = [0]  # PAD only
        inp = torch.tensor([indices], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inp)
            pred = torch.argmax(logits, dim=1).item()
        return pred

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        batch = []
        for ex_words in all_ex_words:
            indices = [self.word_embeddings.word_to_index(w) for w in ex_words if self.word_embeddings.contains(w)]
            if not indices:
                indices = [0]
            batch.append(indices)
        batch_padded = pad_batch(batch, pad_idx=0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch_padded)
            preds = torch.argmax(logits, dim=1).tolist()
        return preds