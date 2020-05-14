'''
This script is used to compute the correct BLEU, CIDEr, ROUGE and METEOR scores 
of a checkpoint on the val and test sets without Teacher Forcing.
'''

import sys
from config import config
sys.path += [config.base_path]

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from src.dataloader import *
from src.utils import *
from src.metrics import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# some path
data_folder = config.dataset_output_path  # folder with data files saved by preprocess.py
data_name = config.dataset_basename  # base name shared by data files

checkpoint = config.model_path + 'best_checkpoint_' + data_name + '.pth.tar'  # model checkpoint
word_map_file = config.dataset_output_path + 'wordmap_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with


# load model
checkpoint = torch.load(checkpoint, map_location = str(device))

decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

caption_model = checkpoint['caption_model']


# load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
vocab_size = len(word_map)

# create ix2word map
rev_word_map = {v: k for k, v in word_map.items()}

# normalization transform
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)


'''
Evaluation

input params:
    beam_size: beam size at which to generate captions for evaluation
               set beam_size = 1 if you want to use greedy search

return: 
    bleu4: BLEU-4 score
'''
def evaluate(beam_size):

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(
            data_folder, data_name, 'test', 
            transform = transforms.Compose([normalize])
        ),
        # TODO: batched beam search
        # therefore, DO NOT use a batch_size greater than 1 - IMPORTANT!
        batch_size = 1, 
        shuffle = True, 
        num_workers = 1, 
        pin_memory = True
    )

    # store ground truth captions and predicted captions (word id) of each image
    # for n images, each of them has one prediction and multiple ground truths (a, b, c...):
    # prediction = [ [pred1], [pred2], ..., [predn] ]
    # ground_truth = [ [ [gt1a], [gt1b], [gt1c] ], ..., [ [gtna], [gtnb] ] ]
    ground_truth = list()
    prediction = list()

    # for each image
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader, desc="Evaluating at beam size " + str(beam_size))):

        # move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # forward encoder
        encoder_out = encoder(image)

        # ground_truth
        img_caps = allcaps[0].tolist()
        img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}], img_caps))  # remove <start> and pads
        ground_truth.append(img_captions)

        # prediction (beam search)
        if caption_model == 'show_tell':
            seq = decoder.beam_search(encoder_out, beam_size, word_map)
        elif caption_model == 'att2all' or caption_model == 'spatial_att':
            seq, _ = decoder.beam_search(encoder_out, beam_size, word_map)
        elif caption_model == 'adaptive_att':
            seq, _, _ = decoder.beam_search(encoder_out, beam_size, word_map)
    
        pred = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        prediction.append(pred)

        assert len(ground_truth) == len(prediction)

    # calculate metrics
    metrics = Metrics(ground_truth, prediction, rev_word_map)
    scores = metrics.all_metrics()

    return scores


if __name__ == '__main__':
    
    beam_size = 5

    (bleu1, bleu2, bleu3, bleu4), cider, rouge, meteor = evaluate(beam_size)

    print("\nScores @ beam size of %d are:" % beam_size)
    print("   BLEU-1: %.4f" % bleu1)
    print("   BLEU-2: %.4f" % bleu2)
    print("   BLEU-3: %.4f" % bleu3)
    print("   BLEU-4: %.4f" % bleu4)
    print("   CIDEr: %.4f" % cider)
    print("   ROUGE-L: %.4f" % rouge)
    print("   METEOR: %.4f" % meteor)