# decoder for paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML 2015.
# with attention

import torch
from torch import nn
import torchvision
from .decoder import Decoder as BasicDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class Attention(): soft attention network (in decoder)

input param:
    encoder_dim: feature size of encoded images
    decoder_dim: dimention of decoder's hidden layer
    attention_dim: dimention of attention network
'''
class Attention(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)  # softmax layer to calculate weights

    '''
    input param:
        encoder_out: feature map extracted by encoder (batch_size, num_pixels, encoder_dim)
        decoder_hidden: previous hidden state (h_{t-1}) of decoder (batch_size, decoder_dim)
    return: 
        attention_weighted_encoding: weighted feature vector (batch_size, encoder_dim)
        alpha: attention weight α_t (batch_size, num_pixels)
    '''
    def forward(self, encoder_out, decoder_hidden):
        # W_a * A
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        # W_g * h_{t-1}
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # e_t = W_e * ReLU(W_a * A + W_g * h_{t-1} * 1^T)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # α_t = softmax(e_t)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        # \sum_i^L(α_{t,i} * a_i)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


'''
class Decoder(): decoder with attention

input param:
    attention_dim: dimention of attention network
    embed_dim: dimention of word embeddings
    decoder_dim: dimention of decoder's hidden layer
    vocab_size: size of vocab vocabulary
    encoder_dim: feature size of encoded images
    dropout: dropout
'''
class Decoder(BasicDecoder):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim = 2048, dropout = 0.5):
        super(Decoder, self).__init__(
            embed_dim = embed_dim, 
            decoder_dim = decoder_dim, 
            vocab_size = vocab_size,
            dropout = dropout
        )

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias = True)  # LSTM
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # layer to initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # layer to initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()


    '''
    initialize cell state and hidden state for LSTM (based on feature map)

    input param:
        encoder_out: feature map extracted by encoder (batch_size, num_pixels, encoder_dim)
    return: 
        h: intial hidden state (batch_size, decoder_dim)
        c: intial cell state (batch_size, decoder_dim)
    '''
    def init_hidden_state(self, encoder_out):
        # 1/L * (\sum_i^L a_i)
        mean_encoder_out = encoder_out.mean(dim = 1)
        h = self.init_h(mean_encoder_out) # h_0: (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out) # c_0: (batch_size, decoder_dim)
        return h, c

    '''
    input param:
        encoder_out: feature map extracted by encoder (batch_size, num_pixels, encoder_dim)
        encoded_captions: caption after one-hot encoding (batch_size, max_caption_length)
        caption_lengths: caption length (batch_size, 1)
    
    return: 
        predictions: word probability over vocabulary predicted by model
        encoded_captions: sorted encoded captions
        decode lengths: actual caption length - 1
        alphas: attention weights α
        sort indices
    '''
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

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

        # create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # start decoding
        for t in range(max(decode_lengths)):
            # create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. 
            # note that we cannot use 'pack_padded_sequence' provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # attention
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            # a gating variable which decide how much context info will be used at each time step
            # β = sigmod(f_beta(h_{t-1}))
            beta = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar (batch_size_t, encoder_dim)
            # z_t = β * \sum_i^L(α_{t,i} * a_i)
            attention_weighted_encoding = beta * attention_weighted_encoding
            
            # LSTM
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim = 1),
                (h[:batch_size_t], c[:batch_size_t])
            )  # (batch_size_t, decoder_dim)
            
            # calc word probability over vocabulary
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind