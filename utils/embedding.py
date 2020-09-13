import torch
import numpy as np
from tqdm import tqdm

'''
fills embedding tensor with values from the uniform distribution

input params:
    embeddings: embedding tensor
'''
def init_embeddings(embeddings):

    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


'''
creates an embedding tensor for the specified word map, for loading into the model

input params:
    emb_file: file containing embeddings (stored in GloVe format)
    word_map: word map

return: 
    embeddings in the same order as the words in the word map, dimension of embeddings
'''
def load_embeddings(emb_file, word_map):

    # find embedding dimension
    with open(emb_file, 'r') as f:
        embed_dim = len(f.readline().split(' ')) - 1
        num_lines = len(f.readlines())

    vocab = set(word_map.keys())

    # create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), embed_dim)
    init_embeddings(embeddings)

    # read embedding file
    for line in tqdm(open(emb_file, 'r'), total = num_lines, desc = 'Loading embeddings'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, embed_dim