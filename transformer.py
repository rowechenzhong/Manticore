import torch
from attention import Attention


class Transformer(torch.nn.Module):
    def __init__(self, size, size_internal=None, decoder=True):
        """
        Alright, let's create a Transformer layer.

        :param size: The size of the input and output.
        :param size_internal: The size of the linear layer in the middle.
        :param decoder: Whether this is a decoder layer or not.
        """
        super().__init__()

        if size_internal is None:
            size_internal = size * 4
            # People usually use 4 times the size for the internal size.

        self.attention = Attention(size, decoder=decoder)

        # We'll use 2 linear layers to transform the output of the attention
        # layer into the final output.
        self.linear_1 = torch.nn.Linear(size, size_internal)
        self.linear_2 = torch.nn.Linear(size_internal, size)

        # We also add two LayerNorms. This is a normalization technique
        # that helps with training.
        self.norm_1 = torch.nn.LayerNorm(size)
        self.norm_2 = torch.nn.LayerNorm(size)

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the Transformer layer.

        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, size)
        """
        # First, we need to get the queries, keys, and values.
        query, key, value = self.attention.get_qkv(
            input)  # All (batch_size, seq_len, size)

        # Next, we need to get the attention weights.
        attention_weights = self.attention.get_attention_weights(
            query, key)  # (batch_size, seq_len, seq_len)

        # Now, we need to apply the attention weights to the values.
        # This will give us the output of the attention layer.
        # (batch_size, seq_len, size)
        attention_output = torch.bmm(attention_weights, value)

        # Apply a LayerNorm to the attention output.
        # (batch_size, seq_len, size)
        attention_output = self.norm_1(attention_output)

        # Finally, we need to apply a linear layer to the attention output
        # to get the final output. Note that we also add the attention output
        # to the linear output. This is called a residual connection.
        output = self.linear_2(torch.nn.functional.gelu(
            self.linear_1(attention_output))) + attention_output  # (batch_size, seq_len, size)

        # Apply a LayerNorm to the output.
        output = self.norm_2(output)
        return output
