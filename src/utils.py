import numpy as np
import torch
import yaml
from config import config

'''
fills embedding tensor with values from the uniform distribution

input param:
    embeddings: embedding tensor
'''
def init_embedding(embeddings):

    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


'''
creates an embedding tensor for the specified word map, for loading into the model

input param:
    emb_file: file containing embeddings (stored in GloVe format)
    word_map: word map

return: 
    embeddings in the same order as the words in the word map, dimension of embeddings
'''
def load_embeddings(emb_file, word_map):

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


'''
clips gradients computed during backpropagation to avoid explosion of gradients

input param:
    optimizer: optimizer with the gradients to be clipped
    grad_clip: clip value
'''
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


'''
save model checkpoint

input param:
    epoch: epoch number the current checkpoint have been trained for
    epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    encoder: encoder model
    decoder: decoder model
    encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    decoder_optimizer: optimizer to update decoder's weights
    caption_model
    bleu4: validation BLEU-4 score for this epoch
    is_best: is this checkpoint the best so far?
'''
def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, caption_model, bleu4, is_best):
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
    filename = 'checkpoint_' + config.dataset_basename + '.pth.tar'
    torch.save(state, config.model_path + filename)
    
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, config.model_path + 'best_' + filename)


'''
load a checkpoint, so that we can continue to train on it

input param:
    checkpoint_path: path of the checkpoint
    fine_tune_encoder: fine-tune encoder or not
    encoder_lr: learning rate of encoder (if fine-tune)

return ():
    encoder: encoder model
    decoder: decoder model
    encoder_optimizer: optimizer to update encoder's weights ('none' if there is no optimizer for encoder in checkpoint)
    decoder_optimizer: optimizer to update decoder's weights
    start_epoch: we should start training the model from __th epoch
    epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    best_bleu4: BLEU-4 score of checkpoint
    caption_model
'''
def load_checkpoint(checkpoint_path, fine_tune_encoder, encoder_lr):

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
        encoder_optimizer = torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, encoder.CNN.parameters()),
            lr = encoder_lr
        )
    
    caption_model = checkpoint['caption_model']
        
    return encoder, encoder_optimizer, decoder, decoder_optimizer, \
            start_epoch, epochs_since_improvement, best_bleu4, caption_model


'''
keeps track of most recent, average, sum, and count of a metric
'''
class AverageMeter(object):

    def __init__(self):
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


'''
shrinks learning rate by a specified factor

input param:
    optimizer: optimizer whose learning rate must be shrunk
    shrink_factor: factor in interval (0, 1) to multiply learning rate with
'''
def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


'''
computes top-k accuracy, from predicted and true labels

input param:
    scores: scores from the model
    targets: true labels
    k: k in top-k accuracy

return: 
    top-k accuracy
'''
def accuracy(scores, targets, k):

    batch_size = targets.size(0)
    # Return the indices of the top-k elements along the first dimension (along every row of a 2D Tensor), sorted
    _, ind = scores.topk(k, 1, True, True)
    # The target tensor is the same for each of the top-k predictions (words). Therefore, we need to expand it to  
    # the same shape as the tensor (ind)
    # (double every label in the row --> so every row will contain k elements/k columns) 
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    # Sum up the correct predictions --> we will now have one value (the sum)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    # Devide by the batch_size and return the percentage
    return correct_total.item() * (100.0 / batch_size)