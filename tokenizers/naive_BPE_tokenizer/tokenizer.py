from typing import List
"""
We want to implement BPE (Byte Pair Encoding) tokenization. This is a
subword tokenization technique that is used in NLP. The idea is that we
start with a vocabulary that contains all of the characters in the
language. Then, we find the most common pair of characters in the
vocabulary and merge them into a single character. We repeat this
process until we reach the desired vocabulary size.
"""

# This will load the tokens from the vocab file.


class Tokenizer:
    def __init__(self):
        """
        Alright, let's create a BPE tokenizer.
        """
        self.vocab: List[str] = []
        self.max_token_length = 0

    def load(self, vocab_file, delimiter="<BRUH>", debug=False):
        with open(vocab_file) as f:
            all_text = f.read()
            for token in all_text.split(delimiter):
                self.vocab.append(token)
                self.max_token_length = max(self.max_token_length, len(token))
        if debug:
            length_data = [0 for i in range(self.max_token_length + 1)]
            for token in self.vocab:
                length_data[len(token)] += 1

            print("Loaded vocab of size ", len(self.vocab))
            print("Max token length is ", self.max_token_length)
            for i in range(self.max_token_length + 1):
                print(i, length_data[i])

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
                if corpus[ptr:ptr + self.max_token_length + 1].startswith(self.vocab[i]):
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
