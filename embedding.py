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
        assert size % 2 == 0, "Size must be even."
        # The size must be even for stupid reason: the positional encoding

        # We'll use a simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embedding = torch.nn.Embedding(vocab_size, size)
        self.positional = positional

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the embedding layer.

        input: (batch_size, seq_len) # These are indices
        output: (batch_size, seq_len, size)
        """
        # First, we need to get the embeddings.
        # Note that we multiply the embedding matrix by the sqrt of the size.
        # This is because we want the variance of the embeddings to be 1.

        # check dtype of input
        # print("In embedding forward")
        # print(input.dtype)

        embedding = self.embedding(input) * np.sqrt(self.size)

        # Next, we need to add the positional encoding.
        # Note that we expand the batch size to be the same as the number of
        # positions.
        if self.positional:
            seq_len = input.shape[1]
            x = torch.arange(seq_len).unsqueeze(1).repeat(1, self.size // 2)
            positional_encoding = torch.zeros_like(embedding)
            period = 10000 ** (2 * torch.arange(self.size // 2) / self.size)

            # print("Periods: ", period)
            # Periods:
            # tensor([1.0000e+00, 1.3335e+00, 1.7783e+00, 2.3714e+00, 3.1623e+00, 4.2170e+00,
            # 5.6234e+00, 7.4989e+00, 1.0000e+01, 1.3335e+01, 1.7783e+01, 2.3714e+01,
            # 3.1623e+01, 4.2170e+01, 5.6234e+01, 7.4989e+01, 1.0000e+02, 1.3335e+02,
            # 1.7783e+02, 2.3714e+02, 3.1623e+02, 4.2170e+02, 5.6234e+02, 7.4989e+02,
            # 1.0000e+03, 1.3335e+03, 1.7783e+03, 2.3714e+03, 3.1623e+03, 4.2170e+03,
            # 5.6234e+03, 7.4989e+03])

            period = period.unsqueeze(0).repeat(seq_len, 1)
            positional_encoding[:, :, 0::2] = torch.sin(x / period)
            positional_encoding[:, :, 1::2] = torch.cos(x / period)

            embedding += positional_encoding

        return embedding


class UnEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, size):
        """
        UnEmbedding is just a linear layer with a log softmax.
        """
        super().__init__()

        self.size = size
        self.linear = torch.nn.Linear(size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the embedding layer.

        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, vocab_size) # These are log probabilities
        """
        # First, we need to apply a linear layer to the embedding
        # to get the final output.
        # print("In unembedding forward")
        # print(input.shape)
        output = self.linear(input)

        # print("In unembedding forward after linear")
        # print(output.shape)
        # Finally, we need to apply a softmax to the output.
        output = self.softmax(output)

        # print("In unembedding forward after softmax")
        # print(output.shape)
        return output
