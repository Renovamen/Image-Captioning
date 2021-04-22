from abc import ABC, abstractmethod
import torch
from torch import nn
import torchvision

class Decoder(ABC, nn.Module):
    """
    Base class for all decoder

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
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.set_embeddings(embeddings, fine_tune)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)  # layer to compute word probability over vocabulary
        self.init_weights()  # initialize embedding and fc layer with the uniform distribution

    def set_embeddings(self, embeddings: torch.Tensor, fine_tune: bool = True) -> None:
        """
        Set weights of embedding layer

        Parameters
        ----------
        embeddings : torch.Tensor
            Word embeddings

        fine_tune : bool, optional, default=True
            Allow fine-tuning of embedding layer? (only makes sense when using
            pre-trained embeddings)
        """
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embedding.weight = nn.Parameter(embeddings, requires_grad=fine_tune)

    def init_weights(self) -> None:
        """
        Initialize embedding and fc layer with the uniform distribution, bias = 0
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    @abstractmethod
    def init_hidden_state(self, encoder_out: torch.Tensor):
        """
        Initialize cell state and hidden state for LSTM, should be implemented
        by child class.

        Parameters
        ----------
        encoder_out : torch.Tensor
            encoder's output
        """
        pass

    @abstractmethod
    def forward(
        self,
        encoder_out,
        encoded_captions: torch.Tensor,
        caption_lengths: torch.Tensor
    ):
        """
        Forward the decoder, should be implemented by child class

        Parameters
        ----------
        encoder_out : tuple
            encoder's output

        encoded_captions : torch.Tensor (batch_size, max_caption_length)
            Caption after one-hot encoding

        caption_lengths : torch.Tensor (batch_size, 1)
            Caption length
        """
        pass
