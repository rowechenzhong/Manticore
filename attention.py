import torch
import numpy as np


class Attention(torch.nn.Module):
    def __init__(self, size: int,
                 attention_size: int = None,
                 output_size: int = None,
                 heads: int = 1,
                 decoder: bool = True):
        """
        Alright, let's create an attention layer.
        This layer will take in the query and the key-value pairs
        and return the attention weights.

        :param size: The size of the input and output.
        :param attention_size: The size of the Q, K, and V in the attention
        :param heads: The number of heads in the attention
        :param decoder: Whether this is a decoder layer or not.
        """
        super().__init__()
        if attention_size is None:
            attention_size = size * 4

        if output_size is None:
            output_size = size

        self.size: int = size
        self.attention_size: int = attention_size
        self.heads: int = heads
        self.output_size: int = output_size

        # First up, we need to create the queries, keys, and values from
        # the input. We'll use a linear layer to do this.
        self.query = torch.nn.Linear(size, attention_size * heads)
        # A bias is not needed for the key. We will also use shared keys.
        self.key = torch.nn.Linear(size, attention_size, bias=False)
        self.value = torch.nn.Linear(size, output_size * heads)

        self.decoder = decoder

    def get_qkv(self, input):
        """
        This function takes in the input and returns the queries, keys,
        and values.

        :param input: (batch_size, seq_len, size)
        :param query: (batch_size, seq_len, attention_size * heads)
        :param key: (batch_size, seq_len, attention_size)
        :return value: (batch_size, seq_len, output_size * heads)
        """
        # First, we'll pass the input through the linear layers.
        # (batch_size, seq_len, attention_size * heads)
        query = self.query(input)
        key = self.key(input)  # (batch_size, seq_len, attention_size)
        value = self.value(input)  # (batch_size, seq_len, output_size * heads)

        return query, key, value

    def get_attention_weights(self, query: torch.Tensor, key: torch.Tensor):
        """
        This function takes in the queries and keys, and returns
        the attention weights.

        :param query: (batch_size, seq_len, attention_size * heads)
        :param key: (batch_size, seq_len, attention_size)
        :return attention_weights: (batch_size, heads, seq_len, seq_len)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # First, we need to get the dot product of the query and the key.
        # This will give us the unnormalized attention weights; we divide by the
        # square root of the sequence length to normalize.
        # Note that we need to transpose the key.

        query = query.view(batch_size, seq_len, self.heads,
                           self.attention_size).transpose(1, 2)
        # query: (batch_size, heads, seq_len, attention_size)

        # print("Unrolled Query Shape = ", query.shape)

        key = key.view(batch_size, 1, seq_len,
                       self.attention_size).transpose(2, 3)
        # key: (batch_size, 1, attention_size, seq_len)

        # print("Unrolled Key Shape = ", key.shape)

        attention_weights = torch.matmul(query, key) / np.sqrt(seq_len)
        # attention_weights: (batch_size, heads, seq_len, seq_len)

        # print("Normalized attention weights inside model")
        # print(self.size, self.attention_size)
        # print(attention_weights)

        # print("Attention weights shape = ", attention_weights.shape)

        if self.decoder:
            # We need to mask the attention weights for the decoder.
            # This is because the decoder should only be able to attend to
            # the previous positions.
            indices = torch.triu_indices(seq_len, seq_len, offset=1)
            attention_weights[:, :, indices[0], indices[1]] = float('-inf')

        # Now, we need to normalize the attention weights.
        attention_weights = torch.nn.functional.softmax(
            attention_weights, dim=3)

        # print("Softmax attention weights inside model")
        # print(attention_weights)

        return attention_weights

    def get_attention(self, attention_weights: torch.Tensor, value: torch.Tensor):
        """
        This function takes in the attention weights and the values and
        returns the attention.

        attention_weights: (batch_size, heads, seq_len, seq_len)
        value: (batch_size, seq_len, output_size * heads)
        attention: (batch_size, seq_len, heads * output_size)
        """
        batch_size, seq_len, _ = value.shape
        value = value.view(batch_size, seq_len, self.heads,
                           self.output_size).permute(0, 2, 1, 3)
        # value: (batch_size, heads, seq_len, output_size)

        # print("Unrolled value shape = ", value.shape)

        # We need to get the weighted average of the values.
        attention = torch.matmul(attention_weights, value).transpose(1, 2)
        # attention: (batch_size, seq_len, heads, output_size)

        # print("Attention shape = ", attention.shape)
        # print("Batch Size = ", batch_size)
        # print("Seq Len = ", seq_len)
        # print("Heads = ", self.heads)
        # print("Output Size = ", self.output_size)

        # We need to reshape the attention. This may be slow.
        return attention.reshape(batch_size, seq_len, self.heads * self.output_size)

    def forward(self, input):
        """
        This function takes in the input and returns the attention.

        input: (batch_size, seq_len, size)
        attention: (batch_size, seq_len, heads * output_size)
        """

        # print("Input shape = ", input.shape)
        # First, we need to get the queries, keys, and values.
        query, key, value = self.get_qkv(input)

        # print("Query shape = ", query.shape)
        # print("Key shape = ", key.shape)
        # print("Value shape = ", value.shape)

        # Next, we need to get the attention weights.
        attention_weights = self.get_attention_weights(query, key)

        # Finally, we need to get the attention.
        attention = self.get_attention(attention_weights, value)

        return attention


if __name__ == "__main__":
    from tests.test_attention import test_attention, test_attention_fixed
    # assert test_attention_fixed()
    # assert test_attention_fixed(decoder=True)
    assert test_attention()
    # assert test_attention(decoder=True)
    print("All tests passed!")
