from typing import List, Dict, Tuple, Set, Union


class BPE:
    def __init__(self, vocab_size=1000, initial_vocab=None):
        """
        Alright, let's create a BPE tokenizer.
        """
        self.vocab_size: int = vocab_size
        if initial_vocab is None:
            # All unicode symbols.
            initial_vocab = ["<unk>"] + [chr(i) for i in range(256)]
            reverse_initial_vocab = {i: j for j, i in enumerate(initial_vocab)}
        self.initial_vocab: List[str] = initial_vocab
        self.reverse_initial_vocab = reverse_initial_vocab

        self.vocab: List[str] = initial_vocab

    def to_ints(self, corpus: str):
        """
        This function takes in a corpus and returns a list of indices.
        """
        res = []
        for c in corpus:
            if c in self.reverse_initial_vocab:
                res.append(self.reverse_initial_vocab[c])
            else:
                res.append(self.reverse_initial_vocab["<unk>"])
        return res

    def train(self, corpus: List[int]):
        """
        We're going to use the dumbest possible implementation.

        corpus is a list of indices. e.g. "cat" -> [3,1,20]
        """

        while len(self.vocab) < self.vocab_size:
            print(len(self.vocab))
            print("The corpus is length ", len(corpus))
            # First, we need to get the frequencies of all of the pairs of
            # characters in the vocabulary.
            frequencies = [[0 for i in range(len(self.vocab))]
                           for j in range(len(self.vocab))]
            for i in range(len(corpus) - 1):
                frequencies[corpus[i]][corpus[i + 1]] += 1

            # Find largest frequency.
            max_frequency = 0
            max_i = 0
            max_j = 0
            for i in range(len(self.vocab)):
                for j in range(len(self.vocab)):
                    if frequencies[i][j] > max_frequency:
                        max_frequency = frequencies[i][j]
                        max_i = i
                        max_j = j

            # Merge the two characters.
            # e.g., "e" + "t" -> "et"
            self.vocab.append(self.vocab[max_i] + self.vocab[max_j])

            # Replace all occurrences of the two characters with the merged
            # character.
            new_corpus = []
            ptr = 0
            while ptr < len(corpus):
                if ptr < len(corpus) - 1 and corpus[ptr] == max_i and corpus[ptr + 1] == max_j:
                    new_corpus.append(len(self.vocab) - 1)
                    ptr += 2
                else:
                    new_corpus.append(corpus[ptr])
                    ptr += 1
            corpus = new_corpus

        return corpus  # Return the new corpus.

    def tokenize(self, corpus: str):
        """
        This function takes in a corpus and returns the tokenized corpus.
        """

        ptr = 0
        tokenized_corpus = []
        while ptr < len(corpus):
            # We need to find the longest token that matches the corpus.
            # e.g., "cat" -> "ca" + "t"
            # e.g., "caterpillar" -> "ca" + "t" + "erp" + "ill" + "ar"
            token = 0
            longest = 0
            for i in range(len(self.vocab)):
                if corpus[ptr:ptr+50].startswith(self.vocab[i]):
                    if len(self.vocab[i]) > longest:
                        longest = len(self.vocab[i])
                        token = i
            if longest == 0:
                ptr += 1
            else:
                tokenized_corpus.append(token)
                ptr += longest

        return tokenized_corpus

    def detokenize(self, tokenized_corpus: List[int]):
        """
        This function takes in a tokenized corpus and returns the
        detokenized corpus.
        """

        corpus = ""
        for token in tokenized_corpus:
            corpus += self.vocab[token]

        return corpus
