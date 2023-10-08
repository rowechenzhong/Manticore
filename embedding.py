import torch
import numpy as np

# A basic embedding layer to convert one-hot vectors to dense vectors, with positional encoding.


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, size, positional=True):
        """
        Alright, let's create an Embedding layer.
        """
        super().__init__()

        self.size = size

        # We'll use a simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embedding = torch.nn.Embedding(vocab_size, size)
        self.positional = positional

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the embedding layer.

        input: (batch_size, seq_len) # These are 
        output: (batch_size, seq_len, size)
        """
        # First, we need to get the embeddings.
        # Note that we multiply the embedding matrix by the sqrt of the size.
        # This is because we want the variance of the embeddings to be 1.
        embedding = self.embedding(input) * np.sqrt(self.size)

        # Next, we need to add the positional encoding.
        # Note that we expand the batch size to be the same as the number of
        # positions.
        if self.positional:
            seq_len = input.shape[1]
            x = torch.arange(seq_len).expand(
                input.shape[0], seq_len).to(input.device)
            positional_encoding = torch.zeros_like(embedding)
            positional_encoding[:, :, 0::2] = torch.sin(
                x / 10000 ** (2 * torch.arange(self.size // 2) / self.size))
            positional_encoding[:, :, 1::2] = torch.cos(
                x / 10000 ** (2 * torch.arange(self.size // 2) / self.size))

        return embedding
