# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from typing import List


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


class DeepAveragingNetwork(nn.Module):
    """
    Enhanced Deep Averaging Network for sentiment classification.
    Averages word embeddings and passes through deeper feedforward layers.
    """
    def __init__(self, embedding_layer, hidden_size, num_classes=2, dropout=0.3, typo_robust=False):
        super(DeepAveragingNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.hidden_size = hidden_size
        
        if typo_robust:
            # Use a more sophisticated architecture for typo robustness
            self.linear1 = nn.Linear(embedding_layer.embedding_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear3 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout * 0.008)  # Very minimal dropout for maximum learning
        else:
            # Use the optimal architecture for regular dev performance
            self.linear1 = nn.Linear(embedding_layer.embedding_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(dropout * 0.05)  # Minimal dropout for maximum learning
        
        self.relu = nn.ReLU()
        self.typo_robust = typo_robust
        
    def forward(self, x):
        # x is a batch of word indices
        # Get embeddings for all words
        embeddings = self.embedding_layer(x)  # [batch_size, seq_len, embedding_dim]
        
        # Average the embeddings (mean pooling)
        # Handle padding by masking
        mask = (x != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        masked_embeddings = embeddings * mask
        averaged = masked_embeddings.sum(dim=1) / mask.sum(dim=1)  # [batch_size, embedding_dim]
        
        # Pass through feedforward layers based on architecture
        if self.typo_robust:
            # Use 3-layer architecture for typo robustness
            h1 = self.relu(self.linear1(averaged))
            h1 = self.dropout(h1)
            h2 = self.relu(self.linear2(h1))
            h2 = self.dropout(h2)
            output = self.linear3(h2)
        else:
            # Use 2-layer architecture for regular performance
            h1 = self.relu(self.linear1(averaged))
            h1 = self.dropout(h1)
            output = self.linear2(h1)
        
        return output


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, model, word_embeddings, device='cpu'):
        self.model = model
        self.word_embeddings = word_embeddings
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise
        :return: 0 or 1 with the label
        """
        with torch.no_grad():
            # Convert words to indices
            word_indices = []
            for word in ex_words:
                idx = self.word_embeddings.word_indexer.index_of(word)
                if idx == -1:
                    idx = self.word_embeddings.word_indexer.index_of("UNK")
                word_indices.append(idx)
            
            if len(word_indices) == 0:
                # Handle empty sentence
                word_indices = [self.word_embeddings.word_indexer.index_of("UNK")]
            
            # Convert to tensor
            input_tensor = torch.tensor([word_indices], device=self.device)
            
            # Get prediction
            logits = self.model(input_tensor)
            prediction = torch.argmax(logits, dim=1).item()
            
            return prediction


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create embedding layer with fine-tuning enabled
    embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)
    
    # Create model with appropriate architecture
    model = DeepAveragingNetwork(embedding_layer, args.hidden_size, num_classes=2, typo_robust=train_model_for_typo_setting)
    model.to(device)
    
    # Loss and optimizer optimized for maximum performance
    criterion = nn.CrossEntropyLoss()
    
    # Use different settings based on typo setting
    if train_model_for_typo_setting:
        # Use very high LR for typo robustness
        optimal_lr = max(args.lr * 15, 0.015)  # Very high LR for typo robustness
        optimizer = optim.Adam(model.parameters(), lr=optimal_lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    else:
        # Use optimal settings for regular dev performance
        optimal_lr = max(args.lr * 8, 0.008)  # High LR for maximum performance
        optimizer = optim.AdamW(model.parameters(), lr=optimal_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    # Convert examples to tensors
    def examples_to_tensors(exs):
        max_len = max(len(ex.words) for ex in exs) if exs else 1
        X = []
        y = []
        
        for ex in exs:
            word_indices = []
            for word in ex.words:
                idx = word_embeddings.word_indexer.index_of(word)
                if idx == -1:
                    idx = word_embeddings.word_indexer.index_of("UNK")
                word_indices.append(idx)
            
            # Pad sequence
            while len(word_indices) < max_len:
                word_indices.append(0)  # PAD token
            
            X.append(word_indices)
            y.append(ex.label)
        
        return torch.tensor(X, device=device), torch.tensor(y, device=device)
    
    # Training loop optimized for 10 epochs
    model.train()
    best_dev_acc = 0.0
    best_model_state = None
    
    for epoch in range(args.num_epochs):
        # Convert training examples to tensors
        X_train, y_train = examples_to_tensors(train_exs)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Evaluate on dev set every epoch for 10 epochs
        if dev_exs:
            model.eval()
            with torch.no_grad():
                X_dev, y_dev = examples_to_tensors(dev_exs)
                dev_logits = model(X_dev)
                dev_preds = torch.argmax(dev_logits, dim=1)
                dev_acc = (dev_preds == y_dev).float().mean().item()
                
                # Track best model
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_model_state = model.state_dict().copy()
                
                # Print progress
                if epoch % 2 == 0 or epoch == args.num_epochs - 1:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Dev Accuracy: {dev_acc:.4f} (Best: {best_dev_acc:.4f})")
            
            model.train()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return trained classifier
    return NeuralSentimentClassifier(model, word_embeddings, device)

