from config import config
from .show_tell import Decoder as ShowTellDecoder
from .att2all import Decoder as Att2AllDecoder
from .adaptive_att import Decoder as AdaptiveAttDecoder

def setup(vocab_size):

    model_name = config.caption_model
    
    if model_name == 'show_tell':
        model = ShowTellDecoder(
            embed_dim = config.emb_dim,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout
        )
    elif model_name == 'att2all':
        model = Att2AllDecoder(
            attention_dim = config.attention_dim,
            embed_dim = config.emb_dim,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout
        )
    elif model_name == 'adaptive_att' or model_name == 'spatial_att':
        model = AdaptiveAttDecoder(
            attention_dim = config.attention_dim, 
            embed_dim = config.emb_dim,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout,
            caption_model = model_name
        )
    else:
        raise Exception("Model not supported: ", model_name)

    return model