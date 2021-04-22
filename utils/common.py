import torch
from torch import nn, optim
from typing import Tuple

from config import config

def save_checkpoint(
    epoch: int,
    epochs_since_improvement: int,
    encoder: nn.Module,
    decoder: nn.Module,
    encoder_optimizer: optim.Optimizer,
    decoder_optimizer: optim.Optimizer,
    caption_model: str,
    bleu4: float,
    is_best: bool
) -> None:
    """
    Save a model checkpoint

    Parameters
    ----------
    epoch : int
        Epoch number the current checkpoint have been trained for

    epochs_since_improvement : int
        Number of epochs since last improvement in BLEU-4 score

    encoder : nn.Module
        Encoder model

    decoder : nn.Module
        Decoder model

    encoder_optimizer : optim.Optimizer
        Optimizer to update encoder's weights, if fine-tuning

    decoder_optimizer : optim.Optimizer
        Optimizer to update decoder's weights

    caption_model : str
        Type of the caption model

    bleu4 : float
        Validation BLEU-4 score for this epoch

    is_best : bool
        Is this checkpoint the best so far?
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer,
        'caption_model': caption_model
    }
    filename = 'checkpoint_' + config.model_basename + '.pth.tar'
    torch.save(state, config.model_path + filename)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, config.model_path + 'best_' + filename)

def load_checkpoint(
    checkpoint_path: str, fine_tune_encoder: bool, encoder_lr: float
) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.Optimizer, int, int, float, str]:
    """
    Load a checkpoint, so that we can continue to train on it.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint to be loaded

    fine_tune_encoder : bool
        Fine-tune encoder or not

    encoder_lr : float
        Learning rate of encoder (if fine-tune)

    Returns
    -------
    encoder : nn.Module
        Encoder model

    decoder : nn.Module
        Decoder model

    encoder_optimizer : optim.Optimizer
        Optimizer to update encoder's weights ('none' if there is no optimizer
        for encoder in checkpoint)

    decoder_optimizer : optim.Optimizer
        Optimizer to update decoder's weights

    start_epoch : int
        We should start training the model from __th epoch

    epochs_since_improvement : int
        Number of epochs since last improvement in BLEU-4 score

    best_bleu4 : float
        BLEU-4 score of checkpoint

    caption_model : str
        Type of the caption model
    """
    checkpoint = torch.load(checkpoint_path)

    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']

    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']

    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if config.fine_tune_encoder is True and encoder_optimizer is None:
        encoder.CNN.fine_tune(fine_tune_encoder)
        encoder_optimizer = optim.Adam(
            params = filter(lambda p: p.requires_grad, encoder.CNN.parameters()),
            lr = encoder_lr
        )

    caption_model = checkpoint['caption_model']

    return encoder, encoder_optimizer, decoder, decoder_optimizer, \
            start_epoch, epochs_since_improvement, best_bleu4, caption_model

class AverageMeter:
    """
    Keep track of most recent, average, sum, and count of a metric
    """
    def __init__(self, tag = None, writer = None):
        self.writer = writer
        self.tag = tag
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # tensorboard
        if self.writer is not None:
            self.writer.add_scalar(self.tag, val)

def clip_gradient(optimizer: optim.Optimizer, grad_clip: float) -> None:
    """
    Clip gradients computed during backpropagation to avoid explosion of gradients.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer with the gradients to be clipped

    grad_clip : float
        Clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(
    optimizer: optim.Optimizer, shrink_factor: float
) -> None:
    """
    Shrink learning rate by a specified factor.

    Parameters
    ----------
    optimizer : optim.Optimizer
        Optimizer whose learning rate must be shrunk

    shrink_factor : float
        Factor in interval (0, 1) to multiply learning rate with
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def accuracy(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Compute the top-k accuracy.

    Parameters
    ----------
    scores : torch.Tensor
        Scores from the model

    targets : torch.Tensor
        True labels

    k : int
        k in top-k accuracy

    Returns
    -------
    accuracy : float
        Top-k accuracy
    """
    batch_size = targets.size(0)

    # return the indices of the top-k elements along the first dimension (along every row of a 2D Tensor), sorted
    _, ind = scores.topk(k, 1, True, True)

    # The target tensor is the same for each of the top-k predictions (words). Therefore, we need to expand it to
    # the same shape as the tensor (ind)
    # (double every label in the row --> so every row will contain k elements/k columns)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))

    # sum up the correct predictions --> we will now have one value (the sum)
    correct_total = correct.view(-1).float().sum()  # 0D tensor

    # devide by the batch_size and return the percentage
    return correct_total.item() * (100.0 / batch_size)
