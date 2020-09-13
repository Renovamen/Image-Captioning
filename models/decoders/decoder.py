# a template decoder, all decoders should inherit from this class

import torch
from torch import nn
import torchvision

'''
class Decoder(): a template decoder

input params:
    embed_dim: dimention of word embeddings
    embeddings: word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    decoder_dim: dimention of decoder's hidden layer
    vocab_size: size of vocab vocabulary
    dropout: dropout
'''
class Decoder(nn.Module):
    def __init__(self, embed_dim, embeddings, fine_tune, 
                 decoder_dim, vocab_size, dropout = 0.5):

        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.set_embeddings(embeddings, fine_tune)
        
        self.dropout = nn.Dropout(p = self.dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)  # layer to calc word probability over vocabulary
        self.init_weights()  # initialize embedding and fc layer with the uniform distribution

    '''
    set weights of embedding layer

    input param:
        embeddings: word embeddings
        fine_tune: allow fine-tuning of embedding layer? 
                   (only makes sense when using pre-trained embeddings)
    '''
    def set_embeddings(self, embeddings, fine_tune = True):
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embedding.weight = nn.Parameter(embeddings, requires_grad = fine_tune)

    '''
    initialize embedding and fc layer with the uniform distribution, bias = 0
    '''
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    '''
    initialize cell state and hidden state for LSTM
    should be implemented by child class

    input params:
        encoder_out: encoder's output
    '''
    def init_hidden_state(self, encoder_out):
        raise NotImplementedError()

    '''
    forward
    should be implemented by child class

    input params:
        encoder_out: encoder's output
        encoded_captions: caption after one-hot encoding (batch_size, max_caption_length)
        caption_lengths: caption length (batch_size, 1)
    '''
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        raise NotImplementedError()