"""
Implementation of the decoder proposed in paper [1].

References
----------
1. "`Show and Tell: A Neural Image Caption Generator. \
    <https://arxiv.org/abs/1411.4555>`_" Oriol Vinyals, et al. CVPR 2015.
"""

from typing import Tuple, Dict
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

from .decoder import Decoder as BasicDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(BasicDecoder):
    """
    Decoder

    Parameters
    ----------
    embed_dim : int
        Dimention of word embeddings

    embeddings : torch.Tensor
        Word embeddings

    fine_tune : bool
        Allow fine-tuning of embedding layer? (only makes sense when using
        pre-trained embeddings)

    decoder_dim : int
        Dimention of decoder's hidden layer

    vocab_size : int
        Size of vocab vocabulary

    dropout : float, optional, default=0.5
        Dropout
    """

    def __init__(
        self,
        embed_dim: int,
        embeddings: torch.Tensor,
        fine_tune: bool,
        decoder_dim: int,
        vocab_size: int,
        dropout: float = 0.5
    ) -> None:
        super(Decoder, self).__init__(
            embed_dim = embed_dim,
            embeddings = embeddings,
            fine_tune = fine_tune,
            decoder_dim = decoder_dim,
            vocab_size = vocab_size,
            dropout = dropout
        )

        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  # LSTM

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize cell state and hidden state for LSTM (a vector filled with 0)

        Parameters
        ----------
        encoder_out : torch.Tensor (batch_size, embed_dim)
            Image feature extracted by encoder

        Returns
        -------
        h : torch.Tensor (batch_size, decoder_dim)
            Intial hidden state

        c : torch.Tensor (batch_size, decoder_dim)
            Intial cell state
        """
        h = torch.zeros(encoder_out.size(0), self.decoder_dim).to(device)  # h_0: (batch_size, decoder_dim)
        c = torch.zeros(encoder_out.size(0), self.decoder_dim).to(device)  # c_0: (batch_size, decoder_dim)
        return h, c

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        encoder_out : torch.Tensor (batch_size, embed_dim)
            Image feature extracted by encoder

        encoded_captions : torch.Tensor (batch_size, max_caption_length)
            Caption after one-hot encoding

        caption_lengths : torch.Tensor (batch_size, 1)
            Caption length

        Returns
        -------
        predictions : torch.Tensor
            Word probability over vocabulary predicted by model

        encoded_captions : torch.Tensor
            Sorted encoded captions

        decode_lengths : torch.Tensor
            Actual caption length - 1

        sort_ind : torch.Tensor
            Sorted indices
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        # Sort input captions by decreasing lengths
        # Because in 'train.py', 'pack_padded_sequence' will be used to deal with the pads in captions
        # and 'pack_padded_sequence' requires the captions sorted by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # Sort_ind contains elements of the batch index of the tensor encoder_out.
        # For example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0.
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # word embedding
        # each batch contains a caption, all batches have the same number of rows (words),
        # since we previously padded the ones shorter than max_caption_length
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize hidden state and cell state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # we won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # decode lengths = actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # start decoding
        for t in range(max(decode_lengths)):
            # create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep.
            # note that we cannot use 'pack_padded_sequence' provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > t for l in decode_lengths])

            if t == 0:
                # at the first time step, input is image feature
                x_t = encoder_out[:batch_size_t]  # (batch_size_t, embed_dim)
            else:
                # input embeded captions
                x_t = embeddings[:batch_size_t, t - 1, :]  # (batch_size_t, embed_dim)

            # LSTM
            h, c = self.decode_step(x_t, (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # compute word probability over vocabulary
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

    def beam_search(
        self, encoder_out: torch.Tensor, beam_size: int, word_map: Dict[str, int]
    ) -> list:
        """
        Beam search (used in evaluation without Teacher Forcing and inference)

        TODO: batched beam search. Therefore, DO NOT use a batch_size greater
        than 1 - IMPORTANT!

        Parameters
        ----------
        encoder_out : torch.Tensor (1, embed_dim)
            Image feature extracted by encoder

        beam_size : int
            Beam size

        word_map : Dict[str, int]
            Word2id map

        Returns
        -------
        predict : list
            Predicted sentence after beam search [word_id1, ..., word_idn]
        """
        k = beam_size

        # dimention of image feature should be the same as dimention of word embedding
        embed_dim = encoder_out.size(-1)
        assert embed_dim == self.embed_dim

        # check the size of vocabulary
        assert len(word_map) == self.vocab_size
        vocab_size = len(word_map)

        # we'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, embed_dim)  # (k, embed_dim)

        # tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out) # (k, decoder_dim)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            if step == 1:
                # at the first time step, input is image feature
                x = encoder_out # (s, embed_dim)
            else:
                # input embeded captions
                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                x = embeddings # (s, embed_dim)

            # LSTM
            h, c = self.decode_step(x, (h, c)) # (s, decoder_dim)

            # calc word probability over the vocabulary
            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim = 1) # (s, vocab_size)

            # record score
            # (k, 1) will be expanded to (k, vocab_size), then (k, vocab_size) + (s, vocab_size) --> (s, vocab_size)
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # for the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1)  # (s, step+1)

            # which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break

            # proceed with incomplete sequences
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # break if things have been going on too long
            if step > 50:
                complete_seqs.extend(seqs.tolist())
                complete_seqs_scores.extend(top_k_scores)
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # predict sentence
        # predict = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]

        return seq
