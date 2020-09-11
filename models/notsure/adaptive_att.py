'''
Paper authors' code is a little different from their paper. This class tried 
to re-implement the adaptive attention network decribed in paper, but failed. 
There are something wrong with alpha_hat_t and beta_t, and I don't know why ...

Anyway, 'decoders/adaptive_att.py' is a correct re-implementation which refers to paper authors' code.

class AdaptiveAttention(): adaptive attention

input params:
    attention_dim: dimention of attention network
    decoder_dim: dimention of decoder's hidden layer
    dropout: dropout
    caption_model: caption model
'''

class AdaptiveAttention(nn.Module):
    
    def __init__(self, attention_dim, decoder_dim, dropout = 0.5, caption_model = 'adaptive_att'):
        super(AdaptiveAttention, self).__init__()

        self.caption_model = caption_model

        self.affine_s = nn.Linear(decoder_dim, decoder_dim)
        self.affine_h = nn.Linear(decoder_dim, decoder_dim)

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
        self.w_p = nn.Linear(decoder_dim, decoder_dim) # w_p

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)


    '''
    input params:
        V: spatial image feature, V = [ v_1, v_2, ... v_num_pixels ] (batch_size, num_pixels = 49, decoder_dim)
        h_t: hiddent state at time t (batch_size, decoder_dim)
        s_t: visual sentinel at time t (batch_size, decoder_dim)
    return:
        spatial_out/adaptive_out: context vector in spatial/adaptive attention (batch_size, decoder_dim)
        alpha_t/alpha_hat_t: attention weight in spatial/adaptive attention (batch_size, num_pixels)
        beta_t: sentinel gate in adaptive attention (batch_size, 1)
    '''
    def forward(self, V, h_t, s_t):

        s_t = self.relu(self.affine_s(s_t)) # (batch_size, decoder_dim)
        h_t = self.tanh(self.affine_h(h_t)) # (batch_size, decoder_dim)

        # W_v * V
        visual_att = self.w_v(V) # (batch_size, num_pixels = 49, attention_dim)
        # W_g * h_t * 1^T
        hidden_att = self.w_g(h_t).unsqueeze(1) # (batch_size, 1, attention_dim)
        # tanh(W_v * V + W_g * h_t * 1^T)
        att = self.tanh(visual_att + hidden_att)  # (batch_size, num_pixels, attention_dim)
        # eq.6: z_t = w_h * att
        z_t = self.w_h(att).squeeze(2)  # (batch_size, num_pixels)
        # eq.7: α_t = softmax(z_t)
        alpha_t = self.softmax(z_t)  # (batch_size, num_pixels)

        # eq.8: c_t = \sum_i^k α_{ti} v_{ti}
        c_t = (V * alpha_t.unsqueeze(2)).sum(dim = 1) # (batch_size, decoder_dim)

        # w_h * tanh(W_s * s_t + W_g * h_t)
        z_t_extended = self.w_h(self.tanh(self.w_s(s_t) + self.w_g(h_t))) # (batch_size, 1)
        # [z_t; z_t_extended]
        extended = torch.cat((z_t, z_t_extended), dim = 1) # (batch_size, num_pixels + 1)
        # eq.12: \hat{α}_t = softmax([z_t; z_t_extended])
        alpha_hat_t = self.softmax(extended) # (batch_size, num_pixels + 1)

        # β_t = \hat{α}_t[k + 1]
        beta_t = alpha_hat_t[:, -1].unsqueeze(1) # (batch_size, 1)

        # eq.11: \hat{c}_t = β_t * s_t + (1 - β_t) * c_t
        # c_hat_t = beta_t * s_t + (1 - beta_t) * c_t # (batch_size, decoder_dim)

        # \hat{c}_t = β_t * s_t + (1 - β_t) * c_t
        c_hat_t = beta_t * s_t + (1 - beta_t) * c_t # (batch_size, decoder_dim)
        # c_hat_t = beta_t * s_t + beta_t * c_t # (batch_size, decoder_dim)

        # eq.13: W_p(c_hat_t + h_t) 
        spatial_out = self.tanh(self.w_p(c_t + h_t)) # (batch_size, decoder_dim)
        adaptive_out = self.tanh(self.w_p(c_hat_t + h_t)) # (batch_size, decoder_dim)


        if self.caption_model == 'adaptive_att':
            # remember, \hat{α} has been extended to num_pixels + 1 in adaptive attention, and we don't need the last element (used to compute beta) anymore
            return adaptive_out, alpha_hat_t[:, :-1], beta_t
        
        elif self.caption_model == 'spatial_att':
            return spatial_out, alpha_t

        # return spatial_out, adaptive_out, alpha_t, alpha_hat_t, beta_t