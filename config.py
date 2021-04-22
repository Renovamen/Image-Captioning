"""
Define the hyper parameters here.
"""

import os

class config:
    # global parameters
    base_path = os.path.abspath(os.path.dirname(__file__))  # path to this project
    caption_model = 'adaptive_att'  # 'show_tell', 'att2all', 'adaptive_att', 'spatial_att'
                                    # refer to README.md for more info about each model

    # dataset parameters
    dataset_image_path = os.path.join(base_path, 'data/flickr8k/images/')
    dataset_caption_path = os.path.join(base_path, 'data/flickr8k/dataset.json')
    dataset_output_path = os.path.join(base_path, 'data/output/flickr8k/')  # folder with data files saved by preprocess.py
    dataset_basename = 'flickr8k'  # any name you want

    # preprocess parameters
    captions_per_image = 5
    min_word_freq = 5  # words with frenquence lower than this value will be mapped to '<UNK>'
    max_caption_len = 50  # captions with length higher than this value will be ignored,
                          # with length lower than this value will be padded from right side to fit this length

    # word embeddings parameters
    embed_pretrain = True  # false: initialize embedding weights randomly
                           # true: load pre-trained word embeddings
    embed_path = os.path.join(base_path, 'data/glove/glove.6B.300d.txt')  # only makes sense when `embed_pretrain = True`
    embed_dim = 512  # dimension of word embeddings
                     # only makes sense when `embed_pretrain = False`
    fine_tune_embeddings = True  # fine-tune word embeddings?

    # model parameters
    attention_dim = 512  # dimension of attention network
                         # you only need to set this when the model includes an attention network
    decoder_dim = 512  # dimension of decoder's hidden layer
    dropout = 0.5
    model_path = os.path.join(base_path, 'checkpoints/')  # path to save checkpoints
    model_basename = 'adaptive_att_8k'  # any name you want

    # training parameters
    epochs = 20
    batch_size = 80
    fine_tune_encoder = False  # fine-tune encoder or not
    encoder_lr = 1e-4  # learning rate of encoder (if fine-tune)
    decoder_lr = 4e-4  # learning rate of decoder
    grad_clip = 5.  # gradient threshold in clip gradients
    checkpoint = None  # path to load checkpoint, None if none
    workers = 0  # num_workers in dataloader
    tau = 1.  # penalty term τ for doubly stochastic attention in paper: show, attend and tell
              # you only need to set this when 'caption_model' is set to 'att2all'
    # tensorboard
    tensorboard = True  # enable tensorboard or not?
    log_dir = os.path.join(base_path, 'logs/adaptive_att/')  # folder for saving logs for tensorboard
                                                             # only makes sense when `tensorboard = True`
