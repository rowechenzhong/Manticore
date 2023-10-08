import torch
from transformer import Transformer
from attention import Attention


class Model(torch.nn.Module):
    def __init__(self, size=64, layers=8, size_internal=None, decoder=False, position=True):
        """
        Alright, let's create a Transformer layer.
        """

        if size_internal is None:
            size_internal = size * 4
            # People usually use 4 times the size for the internal size.

        transformers = []
        for i in range(layers):
            transformers.append(Transformer(
                size, size_internal, decoder=decoder))
        self.transformers = torch.nn.ModuleList(transformers)
