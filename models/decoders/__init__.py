from config import config
from .show_tell import Decoder as ShowTellDecoder
from .att2all import Decoder as Att2AllDecoder
from .adaptive_att import Decoder as AdaptiveAttDecoder

'''
setup a decoder

input params:
    vocab_size: size of vocabulary
    embed_dim: dimention of word embeddings
    embeddings: word embeddings
'''
def setup(vocab_size, embed_dim, embeddings):

    model_name = config.caption_model
    
    if model_name == 'show_tell':
        model = ShowTellDecoder(
            embed_dim = embed_dim,
            embeddings = embeddings,
            fine_tune = config.fine_tune_embeddings,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout
        )
    elif model_name == 'att2all':
        model = Att2AllDecoder(
            embed_dim = embed_dim,
            embeddings = embeddings,
            fine_tune = config.fine_tune_embeddings,
            attention_dim = config.attention_dim,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout
        )
    elif model_name == 'adaptive_att' or model_name == 'spatial_att':
        model = AdaptiveAttDecoder(
            embed_dim = embed_dim,
            embeddings = embeddings,
            fine_tune = config.fine_tune_embeddings,
            attention_dim = config.attention_dim,
            decoder_dim = config.decoder_dim,
            vocab_size = vocab_size,
            dropout = config.dropout,
            caption_model = model_name
        )
    else:
        raise Exception("Model not supported: ", model_name)

    return model