import json
import numpy as np
from typing import Dict
# from scipy.misc import imread, imresize
from imageio import imread
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import visualize_att_beta, visualize_att

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_caption(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: str,
    word_map: Dict[str, int],
    caption_model: str,
    beam_size: int = 3
):
    """
    Generate a caption on a given image using beam search.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder model

    decoder : torch.nn.Module
        Decoder model

    image_path : str
        Path to image

    word_map : Dict[str, int]
        Word map

    beam_size : int, optional, default=3
        Number of sequences to consider at each decode-step

    return:
        seq: caption
        alphas: weights for visualization
    """

    # read and process an image
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis = 2)
    # img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    # prediction (beam search)
    if caption_model == 'show_tell':
        seq = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq
    elif caption_model == 'att2all' or caption_model == 'spatial_att':
        seq, alphas = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq, alphas
    elif caption_model == 'adaptive_att':
        seq, alphas, betas = decoder.beam_search(encoder_out, beam_size, word_map)
        return seq, alphas, betas


if __name__ == '__main__':
    model_path = 'checkpoints/checkpoint_adaptive_att_8k.pth.tar'
    img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/3247052319_da8aba1983.jpg' # man in a four wheeler
    # img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/127490019_7c5c08cb11.jpg' # woman golfing
    # img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/3238951136_2a99f1a1a8.jpg' # man on rock
    # img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/3287549827_04dec6fb6e.jpg' # snowboarder
    # img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/491405109_798222cfd0.jpg' # girl smiling
    # img = '/Users/zou/Renovamen/Developing/Image-Captioning/data/flickr8k/images/3425835357_204e620a66.jpg' # man handstanding
    wordmap_path = 'data/output/flickr8k/wordmap_flickr8k.json'
    beam_size = 5
    ifsmooth = False

    # load model
    checkpoint = torch.load(model_path, map_location=str(device))

    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()

    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    caption_model = checkpoint['caption_model']

    # load word map (word2ix)
    with open(wordmap_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # encoder-decoder with beam search
    if caption_model == 'show_tell':
        seq = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
        caption = [rev_word_map[ind] for ind in seq if ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        print('Caption: ', ' '.join(caption))

    elif caption_model == 'att2all' or caption_model == 'spatial_att':
        seq, alphas = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
        alphas = torch.FloatTensor(alphas)
        # visualize caption and attention of best sequence
        visualize_att(
            image_path = img,
            seq = seq,
            rev_word_map = rev_word_map,
            alphas = alphas,
            smooth = ifsmooth
        )

    elif caption_model == 'adaptive_att':
        seq, alphas, betas = generate_caption(encoder, decoder, img, word_map, caption_model, beam_size)
        alphas = torch.FloatTensor(alphas)
        visualize_att_beta(
            image_path = img,
            seq = seq,
            rev_word_map = rev_word_map,
            alphas = alphas,
            betas = betas,
            smooth = ifsmooth
        )
