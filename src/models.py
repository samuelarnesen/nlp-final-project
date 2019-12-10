""" models """
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


""" Credit: this code was inspired by pytorch.org/tutorials/beginner/transformer_tutorial.html """

# helper module for our classifier
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """ d_model = word embedding dimension of the transformer inputs """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # add batch dimension
        self.register_buffer('pe', pe) # do not perform gradient descent on positional embeddings!!

    def forward(self, x): # BERT also adds positional encodings directly to embeddings
        x = x + self.pe[:, :x.size(1), :] # only include sentence-length many points
        return self.dropout(x)


# implement classification transformer specific for our application
class TransformerClassifier(nn.Module):

    def __init__(self, vocab_size, labels, embedding_dim, nhead, feedforward_dim, nlayers, dropout=0.5):
        """
        Args:
            vocab_size: number of words/max index of our vocabulary
            labels: number of labels in our predictions
            embedding_dim: word embedding dimension
            nhead: number of attention heads
            feedforward_dim: dimension of feedforward layers
            nlayers: number of attention layers to stack
            dropout: dropout rate
        """
        super(TransformerClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.labels = labels
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # word embedding layer
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout) # positional embedding
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, feedforward_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # transformer
        self.linear_classifier = nn.Linear(embedding_dim, labels) # transformer output to class scores
        self.softmax = nn.Softmax(dim=1) # softmax over scores
        self.init_weights()

    def init_weights(self):
        initrange = 0.1 # small variance has been shown to lead to better embedding initialization
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_classifier.bias.data.zero_()
        initrange = np.sqrt(6) / np.sqrt(self.embedding_dim + self.labels) # glorot initialization
        self.linear_classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """ src must be formatted with dims (batch_size, sentence_length, one_hot_labels) """
        word_embedding = self.embedding(src)
        word_pos_embed = self.pos_encoder(word_embedding)
        encoder_output = (self.transformer_encoder(word_pos_embed)[:,0,:]).squeeze(1) # use only the first word's embedding
        scores = self.linear_classifier(encoder_output)
        softmax_scores = self.softmax(scores) # softmax over scores
        return softmax_scores


# for debugging
if __name__ == "__main__":
    vocab_size = 5
    labels = 6
    embedding_dim = 32
    nhead = 1
    feedforward_dim = 64
    nlayers = 1
    t = TransformerClassifier(vocab_size, labels, embedding_dim, nhead, feedforward_dim, nlayers)
    #out = t.forward(torch.rand(3, 10, vocab_size).long())
    out = t.forward(torch.from_numpy(np.array([[3, 2, 4], [0, 1, 4]])))
    print(out)
    print(out.sum(1)) # batch axis all sums to 1

