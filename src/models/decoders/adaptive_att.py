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
class AdaptiveAttention(): adaptive attention

input param:
    attention_dim: dimention of attention network
    decoder_dim: dimention of decoder's hidden layer
    dropout: dropout
'''
class AdaptiveAttention(nn.Module):

    def __init__(self, attention_dim, decoder_dim, dropout = 0.5, caption_model = 'adaptive_att'):
        super(AdaptiveAttention, self).__init__()
        self.affine_s = nn.Linear(decoder_dim, decoder_dim)
        self.affine_h = nn.Linear(decoder_dim, decoder_dim)
        
        self.w_s = nn.Linear(decoder_dim, attention_dim)   
        self.w_g = nn.Linear(decoder_dim, attention_dim)
        self.w_v = nn.Linear(decoder_dim, attention_dim)
        self.w_h = nn.Linear(attention_dim, 1)
        self.W_p = nn.Linear(decoder_dim, decoder_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)  # softmax layer to calculate weights


    '''
    input param:
        V: spatial image feature, V = [ v_1, v_2, ... v_num_pixels ] (batch_size, num_pixels = 49, decoder_dim)
        h_t: hiddent state at time t (batch_size, decoder_dim)
        s_t: visual sentinel at time t (batch_size, decoder_dim)
    return:
        spatial_out/adaptive_out: context vector in spatial/adaptive attention (batch_size, decoder_dim)
        alpha_t/alpha_hat_t: attention weight in spatial/adaptive attention (batch_size, num_pixels)
        beta_t: sentinel gate in adaptive attention (batch_size, 1)
    '''
    def forward(self, V, h_t, s_t, caption_model):

        s_t = self.relu(self.affine_s(s_t)) # (batch_size, decoder_dim)
        h_t = self.tanh(self.affine_h(h_t)) # (batch_size, decoder_dim)

        # W_g * h_t * 1^T
        hidden_att = self.w_g(h_t).unsqueeze(1)  # (batch_size, 1, attention_dim)
        # W_v * V
        visual_att = self.w_v(V) # (batch_size, num_pixels, attention_dim)
        # W_s * s_t
        sentinel_att = self.w_s(s_t) # (batch_size, attention_dim)

        # --------------- Adaptive Attention ---------------
        if caption_model == 'adaptive_att':
            # [W_v * V; W_s * s_t]
            extended = torch.cat([visual_att, sentinel_att.unsqueeze(1)], dim = 1) # (batch_size, num_pixels + 1, attention_dim)
            #   [z_t; w_h * tanh(W_s * s_t + W_g * h_t)]
            # = [w_h * tanh(W_v * V + W_g * h_t * 1^T); w_h * tanh(W_s * s_t + W_g * h_t)]
            # = w_h * tanh([W_v * V; W_s * s_t] + W_g * [h_t * 1^T; h_t])
            extended = self.tanh(extended + hidden_att) # (batch_size, num_pixels + 1, attention_dim)
            z_t_extended = self.w_h(extended).squeeze(2) # (batch_size, num_pixels + 1)
            # eq.12: \hat{α}_t 
            # = softmax([z_t; w_h * tanh(W_s * s_t + W_g * h_t)])
            # = softmax(z_t_extended)
            alpha_hat_t = self.softmax(z_t_extended) # (batch_size, num_pixels + 1)

            concat_feature = torch.cat([V, s_t.unsqueeze(1)], dim = 1) # (batch_size, num_pixels + 1, decoder_dim)
            c_hat_t = (concat_feature * alpha_hat_t.unsqueeze(2)).sum(dim = 1) # (batch_size, decoder_dim)     
            
            # eq.13: W_p(c_hat_t + h_t) 
            adaptive_out = self.tanh(self.W_p(c_hat_t + h_t))

            # β_t = \hat{α}_t[k + 1]
            beta_t = alpha_hat_t[:, -1].unsqueeze(1) # (batch_size, 1)

            # remember, \hat{α} has been extended to num_pixels + 1 in adaptive attention, and we don't need the last element (used to compute beta) anymore
            return adaptive_out, alpha_hat_t[:, :-1], beta_t

        # --------------- Spatial Attention ---------------
        elif caption_model == 'spatial_att':
            # tanh(W_v * V + W_g * h_t * 1^T)
            att = self.tanh(visual_att + hidden_att) # (batch_size, num_pixels, attention_dim)
            # eq.6: z_t = w_h * att
            z_t = self.w_h(att).squeeze(2) # (batch_size, num_pixels)
            # eq.7: α_t = softmax(z_t)
            alpha_t = self.softmax(z_t) # (batch_size, num_pixels)

            # eq.8: c_t = \sum_i^k α_{ti} v_{ti}
            c_t = (V * alpha_t.unsqueeze(2)).sum(dim = 1) # (batch_size, decoder_dim)
 
            spatial_out = self.tanh(self.W_p(c_t + h_t))

            return spatial_out, alpha_t


class Decoder(BasicDecoder):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, dropout = 0.5, caption_model = 'adaptive_att'):
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
        self.adaptive_attention = AdaptiveAttention(attention_dim, decoder_dim)
        self.caption_model = caption_model

    '''
    initialize cell state and hidden state for LSTM (a vector filled with 0)

    input param:
        spatial_feature: spatial image feature (batch_size, num_pixels, decoder_dim)
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
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        
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

        # create tensors to hold word predicion scores
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
            if self.caption_model == 'adaptive_att':
                att_output, _, _ = self.adaptive_attention(spatial_feature[:batch_size_t], h, s, self.caption_model) # (batch_size_t, decoder_dim)
            elif self.caption_model == 'spatial_att':
                att_output, _ = self.adaptive_attention(spatial_feature[:batch_size_t], h, s, self.caption_model) # (batch_size_t, decoder_dim)

            # calc word probability over vocabulary
            preds = self.fc(self.dropout(att_output)) # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        
        return predictions, encoded_captions, decode_lengths, sort_ind

    
    '''
    beam search (used in evaluation without Teacher Forcing and inference)
    
    TODO: batched beam search
    therefore, DO NOT use a batch_size greater than 1 - IMPORTANT!

    input param:
        encoder_out(tuple): a tuple contains:
            0. spatial_feature: spatial image feature (1, num_pixels, decoder_dim)
            1. global_image: global image feature (1, embed_dim)
        beam_size(int): beam size
        word_map(dict): word2id map
    
    return: 
        seq(list[word_id1, ..., word_idn]): the predicted sentence after beam search
        alphas(list): attention weights at each time step
        betas(list): sentinel gate at each time step
    '''
    def beam_search(self, encoder_out, beam_size, word_map):
        
        import math

        k = beam_size

        spatial_feature, global_image = encoder_out # (1, num_pixels, decoder_dim), (1, embed_dim)
        
        num_pixels = spatial_feature.size(1)
        enc_image_size = int(math.sqrt(num_pixels)) # enc_image_size * enc_image_size = num_pixels
        
        # dimention of spatial image feature should be the same as dimention of decoder's hidden layer
        decoder_dim = spatial_feature.size(-1)
        assert decoder_dim == self.decoder_dim
        
        # check the size of vocabulary
        assert len(word_map) == self.vocab_size
        vocab_size = len(word_map)

        # we'll treat the problem as having a batch size of k
        spatial_feature = spatial_feature.expand(k, num_pixels, decoder_dim)  # (k, num_pixels, decoder_dim)
        
        # tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        # tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        # tensor to store top k sequences' alphas; now they're just 1
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

        # lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_alpha = list()

        if self.caption_model == 'adaptive_att':
            # tensor to store the top k sequences' betas; now they're just 1
            seqs_beta = torch.ones(k, 1, 1).to(device) 
            # lists to store completed sequences' betas
            complete_seqs_beta = list()

        # start decoding
        step = 1
        h, c = self.init_hidden_state(spatial_feature) # (k, decoder_dim)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            
            # concatenate word embeddings and global image features for input to LSTM
            # x_t = [w_t; v^g]
            x = torch.cat((embeddings, global_image.expand_as(embeddings)), dim = 1) # (s, embed_dim * 2)

            # LSTM
            h, c, s = self.decode_step(x, (h, c)) # (s, decoder_dim)
            
            # adaptive attention network
            # spatial_c, adaptive_c, alpha, alpha_hat, beta = self.adaptive_attention(spatial_feature, h, s) # (s, decoder_dim), (s, decoder_dim), (s, num_pixels), (s, num_pixels + 1), (s, 1)
            if self.caption_model == 'adaptive_att':
                att_output, alpha, beta = self.adaptive_attention(spatial_feature, h, s, self.caption_model) # (s, decoder_dim), (s, num_pixels), (s, 1)
            elif self.caption_model == 'spatial_att':
                att_output, alpha = self.adaptive_attention(spatial_feature, h, s, self.caption_model) # (s, decoder_dim), (s, num_pixels)
            
            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            # calc word probability over vocabulary
            scores = self.fc(att_output) # (batch_size, vocab_size)
            scores = F.log_softmax(scores, dim = 1) # (s, vocab_size)
            
            # record score
            # (k, 1) will be expanded to (k, vocab_size), then (k, vocab_size) + (s, vocab_size) --> (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # for the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            
            # add new words, alphas and betas to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1) # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim = 1)  # (s, step+1, enc_image_size, enc_image_size)
            if self.caption_model == 'adaptive_att':
                seqs_beta = torch.cat([seqs_beta[prev_word_inds], beta[prev_word_inds].unsqueeze(1)], dim = 1)  # (s, step + 1, 1)

            # which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            
            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                if self.caption_model == 'adaptive_att':
                    complete_seqs_beta.extend(seqs_beta[complete_inds])   

            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            
            # proceed with incomplete sequences
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            if self.caption_model == 'adaptive_att':
                seqs_beta = seqs_beta[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            spatial_feature = spatial_feature[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # break if things have been going on too long
            if step > 50:
                break
            step += 1
            
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        
        if self.caption_model == 'adaptive_att':
            betas = complete_seqs_beta[i]
            return seq, alphas, betas

        elif self.caption_model == 'spatial_att':
            return seq, alphas