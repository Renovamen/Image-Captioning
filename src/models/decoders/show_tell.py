# decoder for paper: Show and Tell: A Neural Image Caption Generator. CVPR 2015.
# without attention

import torch
from torch import nn
import torchvision
from .decoder import Decoder as BasicDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class Decoder(): decoder

input param:
    embed_dim: dimention of word embeddings
    decoder_dim: dimention of decoder's hidden layer
    vocab_size: size of vocab vocabulary
    dropout: dropout
'''
class Decoder(BasicDecoder):
    def __init__(self, embed_dim, decoder_dim, vocab_size, dropout = 0.5):
        super(Decoder, self).__init__(
            embed_dim = embed_dim, 
            decoder_dim = decoder_dim, 
            vocab_size = vocab_size,
            dropout = dropout
        )

        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias = True)  # LSTM

    '''
    initialize cell state and hidden state for LSTM (a vector filled with 0)

    input param:
        encoder_out: image feature extracted by encoder (batch_size, embed_dim)
    return: 
        h: intial hidden state (batch_size, decoder_dim)
        c: intial cell state (batch_size, decoder_dim)
    '''
    def init_hidden_state(self, encoder_out):
        h = torch.zeros(encoder_out.size(0), self.decoder_dim).to(device) # h_0: (batch_size, decoder_dim)
        c = torch.zeros(encoder_out.size(0), self.decoder_dim).to(device) # c_0: (batch_size, decoder_dim)
        return h, c

    '''
    input param:
        encoder_out: image feature extracted by encoder (batch_size, embed_dim)
        encoded_captions: caption after one-hot encoding (batch_size, max_caption_length)
        caption_lengths: caption length (batch_size, 1)
    
    return: 
        predictions: word probability over vocabulary predicted by model
        encoded_captions: sorted encoded captions
        decode lengths: actual caption length - 1
        sort indices
    '''
    def forward(self, encoder_out, encoded_captions, caption_lengths):

        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # sort input captions by decreasing lengths
        # because in 'train.py', 'pack_padded_sequence' will be used to deal with the pads in captions 
        # and 'pack_padded_sequence' requires the captions sorted by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim = 0, descending = True)
        # sort_ind contains elements of the batch index of the tensor encoder_out.
        # for example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0. 
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # word embedding
        # each batch contains a caption, all batches have the same number of rows (words), 
        # since we previously padded the ones shorter than max_caption_length
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize hidden state and cell state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # decode lengths = actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # start decoding
        for t in range(max(decode_lengths)):
            # create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. 
            # note that we cannot use 'pack_padded_sequence' provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > t for l in decode_lengths])
            
            if t == 0:
                # at the first time step, input is image feature
                x_t = encoder_out[:batch_size_t] # (batch_size_t, embed_dim)
            else:
                # input embeded captions
                x_t = embeddings[:batch_size_t, t, :] # (batch_size_t, embed_dim)
            
            # LSTM
            h, c = self.decode_step(x_t, (h[:batch_size_t], c[:batch_size_t])) # (batch_size_t, decoder_dim) 

            # calc word probability over vocabulary
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind