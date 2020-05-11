'''
CNN: ResNet-101 for encoder
'''

import torch
from torch import nn
import torchvision

'''
class ResNet101(): (pretrained) ResNet-101 network

input param:
    encoded_image_size: size of resized feature map
'''
class ResNet101(nn.Module):
    def __init__(self, encoded_image_size = 7):
        super(ResNet101, self).__init__()
        self.enc_image_size = encoded_image_size # size of resized feature map

        # pretrained ResNet-101 model (on ImageNet)
        resnet = torchvision.models.resnet101(pretrained = True)

        # we need the feature map of the last conv layer,
        # so we remove the last two layers of resnet (average pool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # resize input images with different size to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    '''
    input param:
        images: input image(batch_size, 3, image_size = 256, image_size = 256)
    return: 
        feature_map: feature map after resized (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
    '''
    def forward(self, images):
        feature_map = self.resnet(images) # (batch_size, 2048, image_size/32, image_size/32)
        feature_map = self.adaptive_pool(feature_map) # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        return feature_map

    '''
    input param:
        fine_tune: fine-tune CNN（conv block 2-4）or not
    '''
    def fine_tune(self, fine_tune = True):
        for param in self.resnet.parameters():
            param.requires_grad = False
        # only fine-tune conv block 2-4
        for module in list(self.resnet.children())[5:]:
            for param in module.parameters():
                param.requires_grad = fine_tune


'''
class EncoderResNet(): 
    encoder for paper: Show and Tell: A Neural Image Caption Generator. CVPR 2015.
    without attention

input param:
    encoded_image_size: size of resized feature map
    embed_dim: dimention of the output feature (same as dimension of word embeddings)
'''
class EncoderResNet(nn.Module):
    def __init__(self, encoded_image_size = 7, embed_dim = 512):
        super(EncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)
        self.avg_pool = nn.AvgPool2d(
            kernel_size = encoded_image_size, 
            stride = encoded_image_size
        )
        self.output_layer = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim, momentum = 0.01)
        )

    '''
    input param:
        images: input image (batch_size, 3, image_size = 256, image_size = 256)
    return: 
        out: feature of this image (batch_size, embed_dim = 512)
    '''
    def forward(self, images):
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        batch_size = feature_map.size(0)
        out = self.avg_pool(feature_map).view(batch_size, -1) # (batch_size, 2048)
        out = self.output_layer(out) # (batch_size, embed_dim = 512)
        return out


'''
class AttentionEncoderResNet(): 
    encoder for paper: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML 2015.
    with attention

input param:
    encoded_image_size: size of resized feature map
'''
class AttentionEncoderResNet(nn.Module):
    def __init__(self, encoded_image_size = 7):
        super(AttentionEncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)

    '''
    input param:
        images: input image (batch_size, 3, image_size = 256, image_size = 256)
    return: 
        feature_map: feature map of the image (batch_size, num_pixels = 49, encoder_dim = 2048)
    '''
    def forward(self, images):
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        feature_map = feature_map.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)
        
        batch_size = feature_map.size(0)
        encoder_dim = feature_map.size(-1)
        num_pixels = feature_map.size(1) * feature_map.size(2) # encoded_image_size * encoded_image_size = 49
        
        # flatten image
        feature_map = feature_map.view(batch_size, num_pixels, encoder_dim) # (batch_size, num_pixels = 49, encoder_dim = 2048)
        
        return feature_map


'''
class AdaptiveAttentionEncoderResNet():
    encoder for papaer: Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning. CVPR 2017.
    with attention

input param:
    encoded_image_size: size of resized feature map
    decoder_dim: dimention of spatial image feature (same as dimension of decoder's hidden layer)
    embed_dim: dimention of global image feature (same as dimension of word embeddings)
'''
class AdaptiveAttentionEncoderResNet(nn.Module):
    def __init__(self, encoded_image_size = 7, decoder_dim = 512, embed_dim = 512):
        super(AdaptiveAttentionEncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)
        self.avg_pool = nn.AvgPool2d(
            kernel_size = encoded_image_size, 
            stride = encoded_image_size
        )
        self.global_mapping = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim, momentum = 0.01)
        )
        self.spatial_mapping = nn.Sequential(
            # nn.Dropout(p = 0.5),
            nn.Linear(2048, decoder_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim, momentum = 0.01)
        )

    '''
    input param:
        images: input image (batch_size, 3, image_size = 256, image_size = 256)
    return: 
        feature_map: orignal feature map of the image
        spatial_feature: spatial image feature (batch_size, num_pixels, decoder_dim)
        global_feature: global image feature (batch_size, embed_dim)
    '''
    def forward(self, images):
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        
        batch_size = feature_map.shape[0]
        encoder_dim = feature_map.shape[1] # 2048
        num_pixels = feature_map.shape[2] * feature_map.shape[3] # encoded_image_size * encoded_image_size = 49
    
        global_feature = self.avg_pool(feature_map).view(batch_size, -1) # a^g: (batch_size, 2048)
        # global image feature, eq.16: v^g = ReLU(W_b * a^g)
        global_feature = self.global_mapping(global_feature) # (batch_size, embed_dim = 512)

        feature_map = feature_map.permute(0, 2, 3, 1) # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)
        # A = [ a_1, a_2, ..., a_num_pixels ]
        feature_map = feature_map.view(batch_size, num_pixels, encoder_dim) # (batch_size, num_pixels = 49, 2048)
       
        # spatial image feature: V = [ v_1, v_2, ..., v_num_pixels ]
        # eq.15: v_i = ReLU(W_a * a_i)
        spatial_feature = self.spatial_mapping(feature_map) # (batch_size, num_pixels = 49, decoder_dim = 512)

        # return feature_map, spatial_feature, global_feature
        return (spatial_feature, global_feature)