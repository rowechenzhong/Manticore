import torch
from transformer import Transformer
from embedding import Embedding, UnEmbedding
from tokenizers.tokenizer import Tokenizer


class Model(torch.nn.Module):
    def __init__(self,
                 embedding_in: Embedding,
                 embedding_out: UnEmbedding,
                 transformer_params: dict,
                 layers: int = 8
                 ):
        """
        Alright, let's create a Transformer layer.

        :param embedding_in: The embedding layer for the input.
        :param embedding_out: The embedding layer for the output.
        :param transformer_params: The parameters for the Transformer layer.
        This should include size, size_internal, attention_size, and decoder.
        :param layers: The number of Transformer layers to stack.
        """
        super().__init__()

        self.transformer_params = transformer_params
        transformers = []
        for i in range(layers):
            transformers.append(Transformer(**transformer_params))
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
        # if self.training:
        # First, we need to get the embeddings.
        embedding = self.embedding_in(input)

        # Now, we need to pass the embedding through the Transformer layers.
        for transformer in self.transformers:
            embedding = transformer(embedding)

        # Finally, we need to apply a linear layer to the embedding
        # to get the final output.
        output = self.embedding_out(embedding)

        return output
        # else:
        #     # Auto-regressive decoding.
        #     # We are guaranteed that
        #     embedding = self.embedding_in(input)
