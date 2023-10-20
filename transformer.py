import torch
from attention import Attention
from LayerNorm import LayerNorm


class Transformer(torch.nn.Module):
    def __init__(self, size: int,
                 attention_size: int = None,
                 heads: int = 1,
                 size_internal: int = None,
                 decoder: bool = True):
        """
        Alright, let's create a Transformer layer.

        :param size: The size of the input and output.
        :param attention_size: Used in the attention layer.
        :param heads: Used in the attention layer. Required: size % heads == 0
        :param size_internal: The size of the internal layer.
        :param decoder: Whether this is a decoder layer or not.
        """
        super().__init__()

        if size_internal is None:
            size_internal = size * 4
            # People usually use 4 times the size for the internal size.

        self.attention = Attention(
            size=size,
            attention_size=attention_size,
            output_size=size // heads,
            heads=heads,
            decoder=decoder
        )

        # We'll use 2 linear layers to transform the output of the attention
        # layer into the final output.
        self.linear_1 = torch.nn.Linear(size, size_internal)
        self.linear_2 = torch.nn.Linear(size_internal, size)

        # We also add two LayerNorms. This is a normalization technique
        # that helps with training.
        self.norm_1 = LayerNorm(size)
        self.norm_2 = LayerNorm(size)

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the Transformer layer.

        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, size)
        """

        # First, we'll pass the input through the attention layer.
        # (batch_size, seq_len, size)
        attention_output = self.attention(input) + input

        # Apply a LayerNorm to the attention output.
        # (batch_size, seq_len, size)
        attention_output = self.norm_1(attention_output)

        # Finally, we need to apply a linear layer to the attention output
        # to get the final output.
        # (batch_size, seq_len, size)
        output = self.linear_2(
            torch.nn.functional.gelu(
                self.linear_1(attention_output)
            )
        ) + attention_output

        # Apply a LayerNorm to the output.
        output = self.norm_2(output)
        return output
