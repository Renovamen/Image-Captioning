import os
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import models
from trainer import Trainer
from utils import CaptionDataset, load_embeddings, load_checkpoint
from config import config

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def set_trainer():
    # data parameters
    data_folder = config.dataset_output_path
    data_name = config.dataset_basename

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load word2id map
    word_map_file = os.path.join(data_folder, 'wordmap_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # create id2word map
    rev_word_map = {v: k for k, v in word_map.items()}

    # initialize encoder-decoder framework
    if config.checkpoint is None:
        start_epoch = 0
        epochs_since_improvement = 0
        best_bleu4 = 0.
        caption_model = config.caption_model

        # ------------- word embeddings -------------
        if config.embed_pretrain == True:
            # load pre-trained word embeddings for words in the word map
            embeddings, embed_dim = load_embeddings(
                emb_file = config.embed_path,
                word_map = word_map,
                output_folder = config.dataset_output_path,
                output_basename = config.dataset_basename
            )
        else:
            # or initialize embedding weights randomly
            embeddings = None
            embed_dim = config.embed_dim

        # ----------------- encoder ------------------
        encoder = models.encoders.make(embed_dim=embed_dim)
        encoder.CNN.fine_tune(config.fine_tune_encoder)
        # optimizer for encoder's CNN (if fine-tune)
        if config.fine_tune_encoder:
            encoder_optimizer = optim.Adam(
                params = filter(lambda p: p.requires_grad, encoder.CNN.parameters()),
                lr = config.encoder_lr
            )
        else:
            encoder_optimizer = None

        # ----------------- decoder ------------------
        decoder = models.decoders.make(
            vocab_size = len(word_map),
            embed_dim = embed_dim,
            embeddings = embeddings
        )
        # optimizer for decoder

        # print(len(list(decoder.parameters())))
        decoder_params = list(filter(lambda p: p.requires_grad, decoder.parameters()))
        # print(len(decoder_params))
        if caption_model == 'adaptive_att' or caption_model == 'spatial_att':
            decoder_params = decoder_params + list(encoder.global_mapping.parameters()) \
                                            + list(encoder.spatial_mapping.parameters())
        elif caption_model == 'show_tell':
            decoder_params = decoder_params + list(encoder.output_layer.parameters())

        decoder_optimizer = optim.Adam(
            params = decoder_params,
            lr = config.decoder_lr
        )

    # or load checkpoint
    else:
        encoder,
        encoder_optimizer,
        decoder,
        decoder_optimizer,
        start_epoch,
        epochs_since_improvement,
        best_bleu4
        caption_model = load_checkpoint(config.checkpoint, config.fine_tune_encoder, config.encoder_lr)

    # move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # loss function (cross entropy)
    loss_function = nn.CrossEntropyLoss().to(device)

    # custom dataloaders
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    train_loader = DataLoader(
        CaptionDataset(
            data_folder, data_name, 'train',
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )
    val_loader = DataLoader(
        CaptionDataset(
            data_folder, data_name, 'val',
            transform = transforms.Compose([normalize])
        ),
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.workers,
        pin_memory = True
    )

    trainer = Trainer(
        caption_model = caption_model,
        epochs = config.epochs,
        device = device,
        word_map = word_map,
        rev_word_map = rev_word_map,
        start_epoch = start_epoch,
        epochs_since_improvement = epochs_since_improvement,
        best_bleu4 = best_bleu4,
        train_loader = train_loader,
        val_loader = val_loader,
        encoder = encoder,
        decoder = decoder,
        encoder_optimizer = encoder_optimizer,
        decoder_optimizer = decoder_optimizer,
        loss_function = loss_function,
        grad_clip = config.grad_clip,
        tau = config.tau,
        fine_tune_encoder = config.fine_tune_encoder,
        tensorboard = config.tensorboard,
        log_dir = config.log_dir
    )

    return trainer


if __name__ == '__main__':
    trainer = set_trainer()
    trainer.run_train()
