import torch
from transformer import Transformer
from embedding import Embedding
from tokenizer import Tokenizer


class Model(torch.nn.Module):
    def __init__(self,
                 embedding_in: Embedding,
                 embedding_out: Embedding,
                 size=64,
                 layers=8,
                 size_internal=None,
                 decoder=False
                 ):
        """
        Alright, let's create a Transformer layer.
        """
        super().__init__()

        if size_internal is None:
            size_internal = size * 4
            # People usually use 4 times the size for the internal size.

        transformers = []
        for i in range(layers):
            transformers.append(Transformer(
                size, size_internal, decoder=decoder))
        self.transformers = torch.nn.ModuleList(transformers)

        self.embedding_in = embedding_in
        self.embedding_out = embedding_out

    def forward(self, input):
        """
        This function takes in the input and returns the output after
        applying the Transformer layer.

        input: (batch_size, seq_len, size)
        output: (batch_size, seq_len, size)
        """
        if self.training:
            # First, we need to get the embeddings.
            embedding = self.embedding_in(input)

            # Now, we need to pass the embedding through the Transformer layers.
            for transformer in self.transformers:
                embedding = transformer(embedding)

            # Finally, we need to apply a linear layer to the embedding
            # to get the final output.
            output = self.embedding_out(embedding)

            return output
        else:
            # Auto-regressive decoding.
            # We are guaranteed that
            embedding = self.embedding_in(input)
