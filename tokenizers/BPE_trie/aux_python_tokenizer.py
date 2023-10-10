from typing import List, Dict, Tuple


class BPE:
    def __init__(self, vocab_size: int = 1000, initial_vocab: List[str] = None, max_token_length=50):
        """
        Alright, let's create a BPE tokenizer.
        """
        self.vocab_size: int = vocab_size
        if initial_vocab is None:
            # All unicode symbols.
            initial_vocab = ["<unk>"] + [chr(i) for i in range(256)]
            reverse_initial_vocab = {i: j for j, i in enumerate(initial_vocab)}
        self.initial_vocab: List[str] = initial_vocab[:]
        self.reverse_initial_vocab = reverse_initial_vocab

        self.vocab: List[str] = initial_vocab[:]
        self.vocab_size: List[int] = [len(i) for i in self.vocab]

    def pretokenize(self, corpus: str) -> Dict[Tuple[int], int]:
        """
        We're going to take this corpus, split
        it on whitespace, and return (words, frequencies).

        :param corpus: The corpus to pretokenize.
        :return: words: Dict[Tuple[char], int]
        """
        words: Dict[Tuple[int], int] = {}
        idx: int = 0
        while idx < len(corpus):
            if corpus[idx] == " ":
                idx += 1
            else:
                word: List[int] = []
                while idx < len(corpus) and corpus[idx] != " ":
                    # append the character as an integer.
                    if corpus[idx] not in self.reverse_initial_vocab:
                        word.append(self.reverse_initial_vocab["<unk>"])
                    else:
                        word.append(self.reverse_initial_vocab[corpus[idx]])
                    idx += 1
                word = tuple(word)
                words[word] = words.get(word, 0) + 1

        return words

    def train(self, words: Dict[Tuple[int], int]) -> None:
        """
        We're going to use the dumbest possible implementation.

        :param words: The words to train on, obtained from pretokenize.
        """

        while len(self.vocab) < self.vocab_size:
            print("The vocab is length ", len(self.vocab))
            corpus_length = sum(
                [len(word) * freq for word, freq in words.items()])
            print("The corpus is length ", corpus_length)

            # First, we need to get the frequencies of all of the pairs of
            # characters in the vocabulary.
            frequencies = [[0 for i in range(len(self.vocab))]
                           for j in range(len(self.vocab))]

            for word, freq in words.items():
                for i in range(len(word) - 1):
                    frequencies[word[i]][word[i + 1]] += freq

            # Find largest frequency.
            max_frequency = 0
            max_i = -1
            max_j = -1
            for i in range(len(self.vocab)):
                for j in range(len(self.vocab)):
                    if self.vocab_size[i] + self.vocab_size[j] > self.max_token_length:
                        continue
                    if frequencies[i][j] > max_frequency:
                        max_frequency = frequencies[i][j]
                        max_i = i
                        max_j = j

            if max_i == -1 or max_j == -1:
                print("No more pairs to merge!")
                break

            # Merge the two characters.
            # e.g., "e" + "t" -> "et"
            self.vocab.append(self.vocab[max_i] + self.vocab[max_j])

            # Replace all occurrences of the two characters with the merged
            # character.
            new_words = {}
            for word, freq in words.items():
                new_word = []
                ptr = 0
                while ptr < len(word):
                    if ptr < len(word) - 1 and word[ptr] == max_i and word[ptr + 1] == max_j:
                        new_word.append(len(self.vocab) - 1)
                        ptr += 2
                    else:
                        new_word.append(word[ptr])
                        ptr += 1
                new_words[tuple(new_word)] = freq

            words = new_words
