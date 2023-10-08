import torch
from attention import Attention


class Transformer(torch.nn.Module):
    def __init__(self, size, size_internal=None, decoder=False):
        """
        Alright, let's create a Transformer layer.
        """

        if size_internal is None:
            size_internal = size * 4
            # People usually use 4 times the size for the internal size.

        self.attention = Attention(size, decoder=decoder)

        # We'll use 2 linear layers to transform the output of the attention
        # layer into the final output.
        self.linear_1 = torch.nn.Linear(size, size_internal)
        self.linear_2 = torch.nn.Linear(size_internal, size)

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the Transformer layer.

        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, size)
        """
        # First, we need to get the queries, keys, and values.
        query, key, value = self.attention.get_qkv(input)

        # Next, we need to get the attention weights.
        attention_weights = self.attention.get_attention_weights(query, key)

        # Now, we need to apply the attention weights to the values.
        # This will give us the output of the attention layer.
        attention_output = torch.bmm(attention_weights, value)

        # Finally, we need to apply a linear layer to the attention output
        # to get the final output.
        output = self.linear_2(torch.nn.functional.gelu(
            self.linear_1(attention_output)))

        return output
