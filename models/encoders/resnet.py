"""
Implementation of some encoders using ResNet-101 as backbone.
"""

import torch
from torch import nn
import torchvision
from typing import Tuple

class ResNet101(nn.Module):
    """
    (Pretrained) ResNet-101 network

    Parameters
    ----------
    encoded_image_size : int
        Size of the resized feature map
    """
    def __init__(self, encoded_image_size: int = 7):
        super(ResNet101, self).__init__()
        self.enc_image_size = encoded_image_size  # size of resized feature map

        # pretrained ResNet-101 model (on ImageNet)
        resnet = torchvision.models.resnet101(pretrained = True)

        # we need the feature map of the last conv layer,
        # so we remove the last two layers of resnet (average pool and fc)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # resize input images with different size to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        feature_map : torch.Tensor (batch_size, 2048, encoded_image_size=7, encoded_image_size=7)
            Feature map after resized
        """
        feature_map = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        feature_map = self.adaptive_pool(feature_map)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        return feature_map

    def fine_tune(self, fine_tune: bool = True) -> None:
        """
        Parameters
        ----------
        fine_tune : bool
            Fine-tune CNN (conv block 2-4) or not
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
        # only fine-tune conv block 2-4
        for module in list(self.resnet.children())[5:]:
            for param in module.parameters():
                param.requires_grad = fine_tune


class EncoderResNet(nn.Module):
    """
    Implementation of the encoder proposed in paper [1].

    Parameters
    ----------
    encoded_image_size : int
        Size of resized feature map

    embed_dim : int
        Dimention of the output feature (same as dimension of word embeddings)

    References
    ----------
    1. "`Show and Tell: A Neural Image Caption Generator. \
        <https://arxiv.org/abs/1411.4555>`_" Oriol Vinyals, et al. CVPR 2015.
    """
    def __init__(self, encoded_image_size: int = 7, embed_dim: int = 512) -> None:
        super(EncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)
        self.avg_pool = nn.AvgPool2d(
            kernel_size = encoded_image_size,
            stride = encoded_image_size
        )
        self.output_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim, momentum=0.01)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        out : torch.Tensor (batch_size, embed_dim=512)
            Feature of this image
        """
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        batch_size = feature_map.size(0)
        out = self.avg_pool(feature_map).view(batch_size, -1)  # (batch_size, 2048)
        out = self.output_layer(out)  # (batch_size, embed_dim = 512)
        return out


class AttentionEncoderResNet(nn.Module):
    """
    Implementation of the encoder proposed in paper [1]

    Parameters
    ----------
    encoded_image_size : int
        Size of resized feature map

    References
    ----------
    1. "`Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. \
        <https://arxiv.org/abs/1502.03044>`_" Kelvin Xu, et al. ICML 2015.
    """
    def __init__(self, encoded_image_size: int = 7) -> None:
        super(AttentionEncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images : torch.Tensor (batch_size, 3, image_size=256, image_size=256)
            Input image

        Returns
        -------
        feature_map : torch.Tensor (batch_size, num_pixels=49, encoder_dim=2048)
            Feature map of the image
        """
        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)
        feature_map = feature_map.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)

        batch_size = feature_map.size(0)
        encoder_dim = feature_map.size(-1)
        num_pixels = feature_map.size(1) * feature_map.size(2)  # encoded_image_size * encoded_image_size = 49

        # flatten image
        feature_map = feature_map.view(batch_size, num_pixels, encoder_dim)  # (batch_size, num_pixels = 49, encoder_dim = 2048)

        return feature_map


class AdaptiveAttentionEncoderResNet(nn.Module):
    """
    Implementation of the encoder proposed in paper [1].

    Parameters
    ----------
    encoded_image_size : int
        Size of resized feature map

    decoder_dim : int
        Dimention of spatial image feature (same as dimension of decoder's
        hidden layer)

    embed_dim : int
        Dimention of global image feature (same as dimension of word embeddings)

    References
    ----------
    1. "`Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image \
        Captioning. <https://arxiv.org/abs/1612.01887>`_" Jiasen Lu, et al. CVPR 2017.
    """

    def __init__(
        self,
        encoded_image_size: int = 7,
        decoder_dim: int = 512,
        embed_dim: int = 512
    ) -> None:
        super(AdaptiveAttentionEncoderResNet, self).__init__()
        self.CNN = ResNet101(encoded_image_size)
        self.avg_pool = nn.AvgPool2d(
            kernel_size = encoded_image_size,
            stride = encoded_image_size
        )
        self.global_mapping = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim, momentum=0.01)
        )
        self.spatial_mapping = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(2048, decoder_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(embed_dim, momentum=0.01)
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        images : torch.Tensor
            Input image (batch_size, 3, image_size=256, image_size=256)

        Returns
        -------
        spatial_feature : torch.Tensor (batch_size, num_pixels, decoder_dim)
            Spatial image feature

        global_feature : torch.Tensor (batch_size, embed_dim)
            Global image feature
        """

        feature_map = self.CNN(images)  # (batch_size, 2048, encoded_image_size = 7, encoded_image_size = 7)

        batch_size = feature_map.shape[0]
        encoder_dim = feature_map.shape[1]  # 2048
        num_pixels = feature_map.shape[2] * feature_map.shape[3]  # encoded_image_size * encoded_image_size = 49

        global_feature = self.avg_pool(feature_map).view(batch_size, -1)  # a^g: (batch_size, 2048)
        # global image feature, eq.16: v^g = ReLU(W_b * a^g)
        global_feature = self.global_mapping(global_feature)  # (batch_size, embed_dim = 512)

        feature_map = feature_map.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size = 7, encoded_image_size = 7, 2048)
        # A = [ a_1, a_2, ..., a_num_pixels ]
        feature_map = feature_map.view(batch_size, num_pixels, encoder_dim)  # (batch_size, num_pixels = 49, 2048)

        # spatial image feature: V = [ v_1, v_2, ..., v_num_pixels ]
        # eq.15: v_i = ReLU(W_a * a_i)
        spatial_feature = self.spatial_mapping(feature_map)  # (batch_size, num_pixels = 49, decoder_dim = 512)

        # return feature_map, spatial_feature, global_feature
        return spatial_feature, global_feature
