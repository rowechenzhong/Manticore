import torch


class Attention(torch.nn.Module):
    def __init__(self, size, decoder=False):
        """
        Alright, let's create an attention layer.
        This layer will take in the query and the key-value pairs
        and return the attention weights.

        For simplicity, we will have 1 head in this implementation.
        """

        # First up, we need to create the queries, keys, and values from
        # the input. We'll use a linear layer to do this.
        self.query = torch.nn.Linear(size, size)
        self.key = torch.nn.Linear(size, size)
        self.value = torch.nn.Linear(size, size)

        self.decoder = decoder

    def get_qkv(self, input):
        """
        This function takes in the input and returns the queries, keys,
        and values.

        input: (batch_size, seq_len, size)
        query: (batch_size, seq_len, size)
        key: (batch_size, seq_len, size)
        value: (batch_size, seq_len, size)
        """
        # First, we'll pass the input through the linear layers.
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)

        return query, key, value

    def get_attention_weights(self, query, key):
        """
        This function takes in the queries and keys, and returns
        the attention weights.

        query: (batch_size, seq_len, size)
        key: (batch_size, seq_len, size)
        attention_weights: (batch_size, seq_len, seq_len)
        """
        # First, we need to get the dot product of the query and the key.
        # This will give us the unnormalized attention weights.
        attention_weights = torch.bmm(query, key.transpose(1, 2))

        # To normalize the attention weights, we need to divide by the
        # square root of the size.
        attention_weights = attention_weights / (query.shape[-1] ** 0.5)

        # Now, we need to normalize the attention weights.
        attention_weights = torch.nn.functional.softmax(
            attention_weights, dim=2)

        if self.decoder:
            # We need to mask the attention weights for the decoder.
            # This is because the decoder should only be able to attend to
            # the previous positions.
            batch_size, seq_len, _ = attention_weights.shape
            indices = torch.triu_indices(seq_len, seq_len, offset=1)
            attention_weights[:, indices[0], indices[1]] = float('-inf')

        return attention_weights

    def get_attention(self, attention_weights, value):
        """
        This function takes in the attention weights and the values and
        returns the attention.

        attention_weights: (batch_size, seq_len, seq_len)
        value: (batch_size, seq_len, size)
        attention: (batch_size, seq_len, size)
        """
        # We need to get the weighted average of the values.
        attention = torch.bmm(attention_weights, value)

        # attention: (batch_size, seq_len, size)

        return attention

    def forward(self, input):
        """
        This function takes in the input and returns the attention.

        input: (batch_size, seq_len, size)
        attention: (batch_size, seq_len, size)
        """
        # First, we need to get the queries, keys, and values.
        query, key, value = self.get_qkv(input)

        # query: (batch_size, seq_len, size)
        # key: (batch_size, seq_len, size)
        # value: (batch_size, seq_len, size)

        # Next, we need to get the attention weights.
        attention_weights = self.get_attention_weights(query, key)

        # attention_weights: (batch_size, seq_len, seq_len)

        # Finally, we need to get the attention.
        attention = self.get_attention(attention_weights, value)

        # attention: (batch_size, seq_len, size)

        return attention
