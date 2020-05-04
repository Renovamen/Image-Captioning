import sys
from config import config
sys.path += [config.base_path]

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

import src.models.encoders as Encoder
import src.models.decoders as Decoder
from src.dataset.dataloader import *
from src.utils import *

# global parameters
caption_model = config.caption_model

# data parameters
data_folder = config.dataset_output_path
data_name = config.dataset_basename

# model parameters
emb_dim = config.emb_dim  # 词嵌入向量维度
attention_dim = config.attention_dim  # attention 全连接层维度
decoder_dim = config.decoder_dim  # LSTM 维度
dropout = config.dropout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU / CPU
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# training parameters
epochs = config.epochs
tau = config.tau  # 论文 'Doubly Stochastic Attention' 章中的正则化惩罚项 τ
batch_size = config.batch_size
fine_tune_encoder = config.fine_tune_encoder  # 是否 fine-tune CNN
encoder_lr = config.encoder_lr  # CNN 学习率（如果要 fine-tune 的话）
decoder_lr = config.decoder_lr  # LSTM 学习率
grad_clip = config.grad_clip  # clip gradients 的梯度阈值
checkpoint = config.checkpoint  # checkpoint 路径，没有的话就设 None
workers = config.workers  # torch num_workers

start_epoch = 0
epochs_since_improvement = 0  # 距上一次验证集上的 BLEU 提升过了多少个 epochs
best_bleu4 = 0.  # 当前最高 BLEU-4
print_freq = 100  # print training/validation stats every __ batches


'''
train an epoch

input param:
    train_loader: DataLoader for training data
    encoder: an encoder (based on CNN)
    decoder: a decoder (based on LSTM)
    loss_function: loss_function
    encoder_optimizer: optimizer for encoder (Adam) (if fine-tune)
    decoder_optimizer: optimizer for decoder (Adam)
    epoch: current epoch num
'''
def train(train_loader, encoder, decoder, loss_function, encoder_optimizer, decoder_optimizer, epoch):

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # forward encoder
        imgs = encoder(imgs)

        # forward decoder
        if caption_model == 'att2all':
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        else:
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
  
        # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)[0]

        # calc loss
        loss = loss_function(scores, targets)

        # doubly stochastic attention regularization (in paper: show, attend and tell)
        if caption_model == 'att2all':
            loss += tau * ((1. - alphas.sum(dim = 1)) ** 2).mean()

        # backward
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # print status
        if i % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                        batch_time = batch_time,
                                                                        data_time = data_time, 
                                                                        loss = losses,
                                                                        top5 = top5accs)
            )

'''
validate an epoch (with Tearcher Forcing)

input param:
    val_loader: DataLoader for validation data
    encoder: an encoder (based on CNN)
    decoder: a decoder (based on LSTM)
    loss_function: loss function (cross entropy)

return: 
    bleu4: BLEU-4 score
'''
def validate(val_loader, encoder, decoder, loss_function):

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    ground_truth = list()  # ground_truth (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # forward encoder 
            if encoder is not None:
                imgs = encoder(imgs)
            
            # forward decoder 
            if caption_model == 'att2all':
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            else:
                scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
                
            # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first = True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first = True)[0]

            # calc loss
            loss = loss_function(scores, targets)

            # doubly stochastic attention regularization (in paper: show, attend and tell)
            if caption_model == 'att2all':
                loss += tau * ((1. - alphas.sum(dim = 1)) ** 2).mean()

            # keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), 
                                                                                batch_time = batch_time,
                                                                                loss = losses, 
                                                                                top5 = top5accs)
                )

            # store ground truth captions and predicted captions of each image
            # for n images, each of them has one prediction and multiple ground truths (a, b, c...):
            # hypotheses = [hyp1, hyp2, ..., hypn]
            # ground_truth = [[ref1a, ref1b, ref1c], ..., [refna, refnb]]

            # ground truth
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(
                        lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps
                    )
                )  # remove <start> and pads
                ground_truth.append(img_captions)

            # prediction
            _, preds = torch.max(scores_copy, dim = 2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(ground_truth) == len(hypotheses)

        # calc BLEU-4 score
        bleu4 = corpus_bleu(ground_truth, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss = losses,
                top5 = top5accs,
                bleu = bleu4
            )
        )

    return bleu4


def main():

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # load word2id map
    word_map_file = os.path.join(data_folder, 'wordmap_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # initialize checkpoint
    if checkpoint is None:
        # set up decoder based on chosen model (in 'config.py')
        decoder = Decoder.setup(vocab_size = len(word_map))
        decoder_optimizer = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, decoder.parameters()),
            lr = decoder_lr
        )
        # set up encoder based on chosen model (in 'config.py')
        encoder = Encoder.setup()
        encoder.fine_tune(fine_tune_encoder)
        if fine_tune_encoder:
            encoder_optimizer = torch.optim.Adam(
                params = filter(lambda p: p.requires_grad, encoder.parameters()),
                lr = encoder_lr
            )
        else:
            encoder_optimizer = None
    # load checkpoint
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params = filter(lambda p: p.requires_grad, encoder.parameters()),
                lr = encoder_lr
            )

    # move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # custom dataloaders
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'train', 
            transform = transforms.Compose([normalize])
        ),
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = workers, 
        pin_memory = True
    )
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'val', 
            transform = transforms.Compose([normalize])
        ),
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = workers, 
        pin_memory = True
    )

    # epochs
    for epoch in range(start_epoch, epochs):

        # decay learning rate if there is no improvement for 8 consecutive epochs
        # terminate training if there is no improvement for 20 consecutive epochs
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # train an epoch
        train(
            train_loader = train_loader,
            encoder = encoder,
            decoder = decoder,
            loss_function = loss_function,
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            epoch = epoch
        )

        # validate an epoch
        recent_bleu4 = validate(
            val_loader = val_loader,
            encoder = encoder,
            decoder = decoder,
            loss_function = loss_function
        )

        # epochs num since last improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # save checkpoint
        save_checkpoint(
            data_name = data_name, 
            epoch = epoch, 
            epochs_since_improvement = epochs_since_improvement, 
            encoder = encoder, 
            decoder = decoder, 
            encoder_optimizer = encoder_optimizer,
            decoder_optimizer = decoder_optimizer,
            config = config,
            bleu4 = recent_bleu4, 
            is_best = is_best
        )


if __name__ == '__main__':
    main()