# a template decoder, all decoders should inherit from this class

import torch
from torch import nn
import torchvision

'''
class Decoder(): a template decoder

input param:
    embed_dim: dimention of word embeddings
    decoder_dim: dimention of decoder's hidden layer
    vocab_size: size of vocab vocabulary
    dropout: dropout
'''
class Decoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, dropout = 0.5):

        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # word embedding layer
        self.dropout = nn.Dropout(p = self.dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)  # layer to calc word probability over vocabulary
        self.init_weights()  # initialize embedding and fc layer with the uniform distribution

    '''
    initialize embedding and fc layer with the uniform distribution, bias = 0
    '''
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    '''
    load pretrained word embeddings（optional）

    input param：
        embeddings: pretrained word embeddings
    '''
    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    '''
    fine_tune embedding layer or not（optional, only makes sense when using pretrained embeddings）
    
    input param:
        fine_tune: fine_tune or not
    '''
    def fine_tune_embeddings(self, fine_tune = True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    '''
    initialize cell state and hidden state for LSTM
    should be implemented by child class

    input param:
        encoder_out: encoder's output
    '''
    def init_hidden_state(self, encoder_out):
        raise NotImplementedError()

    '''
    forward
    should be implemented by child class

    input param:
        encoder_out: encoder's output
        encoded_captions: caption after one-hot encoding (batch_size, max_caption_length)
        caption_lengths: caption length (batch_size, 1)
    '''
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        raise NotImplementedError()