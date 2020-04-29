# decoder for paper: Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning. CVPR 2017.
# includes spatial attention and adaptive attention
# with attention

import torch
from torch import nn
import torchvision
from torch.nn import init
import torch.nn.functional as F
from .decoder import Decoder as BasicDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class Sentinel(): calc visual sentinel s_t

input param:
    input_size: dimention of x_t ([ w_t; v^g ] => 2 * embed_dim)
    decoder_dim: dimention of decoder's hidden layer
    dropout: dropout
'''  
class Sentinel(nn.Module):
    def __init__(self, input_size, decoder_dim, dropout = 0.5):
        super(Sentinel, self).__init__()
        self.w_x = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(input_size, decoder_dim)
        ) # W_x
        self.w_h = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(decoder_dim, decoder_dim)
        ) # W_h
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights()
    
    '''
    intalize W_x and W_h
    '''
    def init_weights(self):
        init.xavier_uniform_(self.w_x[1].weight)
        init.xavier_uniform_(self.w_h[1].weight)

    '''
    input param:
        x_t: input ([ w_t; v^g ]) (batch_size, 2 * embed_dim)
        h_last: hiddent state at time t-1 (batch_size, decoder_dim)
        m_t: cell state at time t (batch_size, decoder_dim)
    return:
        s_t: visual sentinel (batch_size, decoder_dim)
    ''' 
    def forward(self, x_t, h_last, m_t):
        # eq.9: g_t = sigmoid(W_x * x_t + W_h * h_{t-1})        
        g_t = self.w_x(x_t) + self.w_h(h_last) # (batch_size, decoder_dim)
        g_t = self.sigmoid(g_t) # (batch_size, decoder_dim)
        # eq.10: s_t = g_t ⊙ tanh(m_t)
        s_t = g_t * self.tanh(m_t) # (batch_size, decoder_dim)
        return s_t


'''
class AdaptiveLSTMCell(): LSTM for adaptive attention (with visual sentinel s_t)

input param:
    input_size: dimention of x_t ([ w_t; v^g ] => 2 * embed_dim)
    decoder_dim: dimention of decoder's hidden layer
''' 
class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, decoder_dim):
        super(AdaptiveLSTMCell, self).__init__()
        self.LSTM = nn.LSTMCell(input_size, decoder_dim, bias = True)
        self.sentinel = Sentinel(input_size, decoder_dim)
    
    '''
    input param:
        x_t: input ([ w_t; v^g ]) (batch_size, 2 * embed_dim)
        states(tuple): a tuple contains:
            0. h_last: hiddent state at time t-1 (batch_size, decoder_dim)
            1. m_last: cell state at time t-1 (batch_size, decoder_dim)
    return:
        h_t: hiddent state at time t (batch_size, decoder_dim)
        m_t: cell state at time t (batch_size, decoder_dim)
        s_t: visual sentinel at time t (batch_size, decoder_dim)
    ''' 
    def forward(self, x_t, states):
        h_last, m_last = states
        h_t, m_t = self.LSTM(x_t, (h_last, m_last))
        s_t = self.sentinel(x_t, h_last, m_t)
        return h_t, m_t, s_t


'''
class AdaptiveAttention(): calc \hat{c}_t

input param:
    attention_dim: dimention of attention network
    decoder_dim: dimention of decoder's hidden layer
    dropout: dropout
'''
class AdaptiveAttention(nn.Module):
    def __init__(self, attention_dim, decoder_dim, dropout = 0.5):
        super(AdaptiveAttention, self).__init__()
        self.w_v = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(decoder_dim, attention_dim)
        ) # W_v
        self.w_g = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(decoder_dim, attention_dim)
        ) # W_g
        self.w_s = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(decoder_dim, attention_dim)
        ) # W_s
        self.w_h = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(attention_dim, 1)
        ) # w_h
        self.softmax = nn.Softmax(dim = 1)  # softmax layer to calculate weights
        self.tanh = nn.Tanh()
        self.init_weights()
    
    '''
    intalize weights
    '''
    def init_weights(self):
        init.xavier_uniform_(self.w_v[1].weight)
        init.xavier_uniform_(self.w_g[1].weight)
        init.xavier_uniform_(self.w_h[1].weight)
        init.xavier_uniform_(self.w_s[1].weight)

    '''
    input param:
        spatial_feature: spatial image feature, V = [ v_1, v_2, ... v_num_pixels ] (batch_size, num_pixels = 49, decoder_dim)
        h_t: hiddent state at time t (batch_size, decoder_dim)
        s_t: visual sentinel at time t (batch_size, decoder_dim)
    return:
        c_t: context vector in spatial attention (batch_size, decoder_dim)
        c_hat_t: context vector in adaptive attention (batch_size, decoder_dim)
        alpha_t: attention weight in spation attention (batch_size, num_pixels)
        beta_t: entinel gate in adaptive attention (batch_size, 1)
    '''
    def forward(self, spatial_feature, h_t, s_t):
        # W_v * V
        visual_att = self.w_v(spatial_feature) # (batch_size, num_pixels = 49, attention_dim)
        # W_g * h_t * 1^T
        hidden_att = self.w_g(h_t).unsqueeze(1) # (batch_size, 1, attention_dim)
        # tanh(W_v * V + W_g * h_t * 1^T)
        att = self.tanh(visual_att + hidden_att)  # (batch_size, num_pixels, attention_dim)
        # eq.6: z_t = w_h * att
        z_t = self.w_h(att).squeeze(2)  # (batch_size, num_pixels)
        # eq.7: α_t = softmax(z_t)
        alpha_t = self.softmax(z_t)  # (batch_size, num_pixels)

        # eq.8: c_t = \sum_i^k α_{ti} v_{ti}
        c_t = (spatial_feature * alpha_t.unsqueeze(2)).sum(dim = 1) # (batch_size, decoder_dim)

        # w_h * tanh(W_s * s_t + W_g * h_t)
        z_t_extended = self.w_h(self.tanh(self.w_s(s_t) + self.w_g(h_t))) # (batch_size, 1)
        # [z_t; z_t_extended]
        extended = torch.cat((z_t, z_t_extended), dim = 1) # (batch_size, num_pixels + 1)
        # eq.12: \hat{α}_t = softmax([z_t; z_t_extended])
        alpha_hat_t = self.softmax(extended) # (batch_size, num_pixels + 1)
        # β_t = \hat{α}_t[k + 1]
        beta_t = alpha_hat_t[:, -1].unsqueeze(1) # (batch_size, 1)

        # eq.11: \hat{c}_t = β_t * s_t + (1 - β_t) * c_t
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t # (batch_size, decoder_dim)

        return c_t, c_hat_t, alpha_t, beta_t

        
class Decoder(BasicDecoder):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, dropout = 0.5):
        super(Decoder, self).__init__(
            embed_dim = embed_dim, 
            decoder_dim = decoder_dim, 
            vocab_size = vocab_size,
            dropout = dropout
        )
        # input is word embedding concatenated with global image feature,
        # so the size of input should be embed_dim * 2 ([ w_t; v^g ] => embed_dim * 2)
        # self.decode_step = nn.LSTMCell(embed_dim * 2, decoder_dim, bias = True)  # LSTM
        self.decode_step = AdaptiveLSTMCell(embed_dim * 2, decoder_dim) # LSTM with visual sentinel
        self.adaptive_attention = AdaptiveAttention(decoder_dim, attention_dim)

    '''
    initialize cell state and hidden state for LSTM (a vector filled with 0)

    input param:
        encoder_out: spatial image feature extracted by encoder (batch_size, num_pixels, decoder_dim)
    return: 
        h: intial hidden state (batch_size, decoder_dim)
        c: intial cell state (batch_size, decoder_dim)
    '''
    def init_hidden_state(self, spatial_feature):
        h = torch.zeros(spatial_feature.size(0), self.decoder_dim).to(device) # h_0: (batch_size, decoder_dim)
        c = torch.zeros(spatial_feature.size(0), self.decoder_dim).to(device) # c_0: (batch_size, decoder_dim)
        return h, c

    '''
    input param:
        encoder_out(tuple): a tuple contains:
            0. spatial_feature: spatial image feature (batch_size, num_pixels, decoder_dim)
            1. global_image: global image feature (batch_size, embed_dim)
        encoded_captions: caption after one-hot encoding (batch_size, max_caption_length)
        caption_lengths: caption length (batch_size, 1)
    
    return: 
        predictions: word probability over vocabulary predicted by model
        encoded_captions: sorted encoded captions
        decode lengths: actual caption length - 1
        sort indices
    '''
    def forward(self, encoder_out, encoded_captions, caption_lengths, caption_model = 'adaptive_att'):
        
        spatial_feature, global_image = encoder_out

        batch_size = spatial_feature.size(0)
        num_pixels = spatial_feature.size(1)
        vocab_size = self.vocab_size

        # sort input captions by decreasing lengths
        # because in 'train.py', 'pack_padded_sequence' will be used to deal with the pads in captions 
        # and 'pack_padded_sequence' requires the captions sorted by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim = 0, descending = True)
        # sort_ind contains elements of the batch index of the tensor encoder_out.
        # for example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0. 
        spatial_feature = spatial_feature[sort_ind]
        global_image = global_image[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # word embedding
        # each batch contains a caption, all batches have the same number of rows (words), 
        # since we previously padded the ones shorter than max_caption_length
        embeddings = self.embedding(encoded_captions) # (batch_size, max_caption_length, embed_dim)

        # initialize hidden state and cell state
        h, c = self.init_hidden_state(spatial_feature) # (batch_size, decoder_dim)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # decode lengths = actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)

        # concatenate word embeddings and global image features for input to LSTM
        # x_t = [w_t; v^g]
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        inputs = torch.cat((embeddings, global_image), dim = 2) # (batch_size, max_caption_length, embed_dim * 2)

        # start decoding
        for t in range(max(decode_lengths)):
            # create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. 
            # note that we cannot use 'pack_padded_sequence' provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > t for l in decode_lengths])
            
            x_t = inputs[:batch_size_t, t, :] # (batch_size_t, embed_dim * 2)
            
            h, c, s = self.decode_step(x_t, (h[:batch_size_t], c[:batch_size_t])) # (batch_size_t, decoder_dim)
            
            # adaptive attention network
            spatial_c, adaptive_c, _, _ = self.adaptive_attention(spatial_feature[:batch_size_t], h, s) # (batch_size_t, decoder_dim)
            
            # calc word probability over vocabulary
            if caption_model == 'adaptive_att':
                # eq.13: p_t = softmax(W_p(\hat{c}_t + h_t))
                preds = self.fc(self.dropout(adaptive_c + h)) # (batch_size, vocab_size)
            elif caption_model == 'spatial_att':
                preds = self.fc(self.dropout(spatial_c + h)) # (batch_size, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        
        return predictions, encoded_captions, decode_lengths, sort_ind