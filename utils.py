from typing import List, Tuple, Dict, Any, Union
import torch
from torch.utils.data import Dataset


class SubstringDataset(Dataset):
    """
    This class is a PyTorch Dataset that returns substrings of a corpus.
    """

    def __init__(self, corpus: torch.Tensor, size: int = 100) -> None:
        """
        Initialize the dataset.
        """
        self.corpus = corpus
        self.size = size

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.corpus) - self.size

    def __getitem__(self, i: int) -> str:
        """
        Return the substring of the corpus at index i.
        """
        return self.corpus[i:i + self.size], self.corpus[i + 1: i + self.size + 1]
