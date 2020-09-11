from config import config
from .resnet import EncoderResNet, AttentionEncoderResNet, AdaptiveAttentionEncoderResNet

def setup():

    model_name = config.caption_model
    
    if model_name == 'show_tell':
        model = EncoderResNet(embed_dim = config.emb_dim)
    elif model_name == 'att2all':
        model = AttentionEncoderResNet()
    elif model_name == 'adaptive_att' or model_name == 'spatial_att':
        model = AdaptiveAttentionEncoderResNet(
            decoder_dim = config.decoder_dim, 
            embed_dim = config.emb_dim
        )
    else:
        raise Exception("Model not supported: ", model_name)

    return model