from typing import List, Tuple, Dict

MAX_TOKEN = 10


class TrieNode:
    def __init__(self, mark=-1):
        """
        Initialize a TrieNode.

        Args:
            mark (int, optional): Mark to identify tokens in the vocabulary. Defaults to -1.
        """
        self.children = {}
        self.mark = mark

    def __del__(self):
        """
        Destructor to delete all children.
        """
        for _, (_, child) in self.children.items():
            del child

    def insert_vector(self, tokens: List[int], pos: int, lim: int, wt: int, leaf_mark=-1) -> 'TrieNode':
        """
        Insert the given indices from a vector from [pos, pos + lim) with weight wt.

        Args:
            tokens (List[int]): List of tokens.
            pos (int): Starting position in the list.
            lim (int): Limit of tokens to insert.
            wt (int): Weight of the tokens.
            leaf_mark (int, optional): Mark for leaf node. Defaults to -1.

        Returns:
            TrieNode: Updated TrieNode.
        """
        temp = self
        for i in range(pos, min(len(tokens), pos + lim)):
            t = tokens[i]
            if t not in temp.children:
                temp.children[t] = (wt, TrieNode())
            else:
                temp.children[t] = (temp.children[t][0] +
                                    wt, temp.children[t][1])
            temp = temp.children[t][1]
        temp.mark = leaf_mark
        return self

    def insert_suffixes(self, tokens: List[int], lim: int, wt: int) -> 'TrieNode':
        """
        Insert all clipped suffixes of the given word with weight wt.

        Args:
            tokens (List[int]): List of tokens.
            lim (int): Limit of tokens to insert.
            wt (int): Weight of the tokens.

        Returns:
            TrieNode: Updated TrieNode.
        """
        for i in range(len(tokens)):
            self.insert_vector(tokens, i, lim, wt)
        return self

    @classmethod
    def create_vocab_trie(cls, vocab_list: List[str]) -> 'TrieNode':
        """
        Create a trie for the vocabulary.

        Args:
            vocab (List[str]): List of vocabulary words.

        Returns:
            TrieNode: Vocabulary trie.
        """
        vocab_trie = TrieNode()
        for idx, tk in enumerate(vocab_list):
            vectorize = [ord(c) for c in tk]
            vocab_trie.insert_vector(vectorize, 0, MAX_TOKEN, 1, idx)
        return vocab_trie


# This will load the tokens from the vocab file.


class Tokenizer:
    def __init__(self):
        """
        Alright, let's create a BPE tokenizer.
        """
        self.vocab: TrieNode = TrieNode()
        self.vocab_list: List[str] = []
        self.max_token_length = MAX_TOKEN

    def __len__(self):
        return len(self.vocab_list)

    def load(self, vocab_file, delimiter="<BRUH>", debug=False):
        self.vocab_list = []
        # lol whoops, add all of the tokens manually
        for i in range(256):
            s = chr(i)
            self.vocab_list.append(s)
        with open(vocab_file, "rb") as f:
            # read f as a list of bytes
            all_text = f.read()
            # convert each individual byte to a string
            all_text = [chr(i) for i in all_text]
            # join all the strings together
            all_text = "".join(all_text)
            for token in all_text.split(delimiter):
                if len(token) <= 1:
                    continue  # we just added the single characters
                self.vocab_list.append(token)
                self.max_token_length = max(self.max_token_length, len(token))
        if debug:
            length_data = [0 for i in range(self.max_token_length + 1)]
            for token in self.vocab_list:
                length_data[len(token)] += 1

            print("Loaded vocab of size ", len(self.vocab_list))
            print("Max token length is ", self.max_token_length)
            for i in range(self.max_token_length + 1):
                print(i, length_data[i])

        self.vocab = TrieNode.create_vocab_trie(self.vocab_list)

    def pre_tokenize(self, corpus: str, reverse_initial_vocab: Dict[str, int]) -> Dict[List[int], int]:
        """
        Perform pre-tokenization.

        Args:
            corpus (str): Input corpus.
            reverse_initial_vocab (Dict[str, int]): Reverse initial vocabulary mapping.

        Returns:
            Dict[List[int], int]: Map of distinct words to their frequency.
        """
        frequencies = {}
        idx = 0
        while idx < len(corpus):
            if idx % 1000 == 0:
                print(
                    f"Pre-tokenizing: {idx} of {len(corpus)} frequency size is {len(frequencies)}", end="\r")
            j = idx
            while j < len(corpus) and corpus[j] not in [' ', '\n', '\t']:
                j += 1
            bytes = [reverse_initial_vocab.get(
                c, reverse_initial_vocab["<unk>"]) for c in corpus[idx:j]]
            frequencies[tuple(bytes)] = frequencies.get(tuple(bytes), 0) + 1
            idx = j + 1
        print()
        return frequencies

    def tokenize(self, corpus: str, debug: bool = False) -> List[int]:
        """
        Tokenize the corpus.

        Args:
            corpus (str): Input corpus.
            vocab_trie (TrieNode): Vocabulary trie.

        Returns:
            List[int]: Tokenized corpus.
        """
        tokenized_corpus = []
        rowechen_ptr = 0
        isaac_ptr = 0
        isaac_notes = 0
        cur: TrieNode = self.vocab

        while True:
            if rowechen_ptr >= len(corpus):
                chr = -1
            else:
                chr = ord(corpus[rowechen_ptr])
            if chr not in cur.children:
                tokenized_corpus.append(isaac_notes)
                rowechen_ptr = isaac_ptr + 1
                if rowechen_ptr >= len(corpus):
                    break
                chr = ord(corpus[rowechen_ptr])
                cur = self.vocab.children[chr][1]
            else:
                cur = cur.children[chr][1]

            if cur.mark != -1:
                isaac_ptr = rowechen_ptr
                isaac_notes = cur.mark

            rowechen_ptr += 1
            if rowechen_ptr & 0xfff == 0:
                print(
                    f"Tokenizing ........ {rowechen_ptr} of {len(corpus)}", end="\r")

        print("Finished tokenizing", end=" " * 29 + "\n")

        if debug:
            # print number of tokens that never got used
            usage = [0 for i in range(len(self.vocab_list))]
            for i in tokenized_corpus:
                usage[i] += 1
            freq_usage = [0 for i in range(10)]
            for i in usage:
                freq_usage[min(i, 9)] += 1
            for i in range(9):
                print(i, freq_usage[i])
            print(">=9", freq_usage[9])

        return tokenized_corpus

    def detokenize(self, corpus: List[int], debug: bool = False) -> str:
        """
        Detokenize the corpus.

        Args:
            corpus (List[int]): Input corpus.

        Returns:
            str: Detokenized corpus.
        """
        detokenized_corpus = ""
        for i in range(len(corpus)):
            if debug and i % 1000 == 0:
                print(f"Detokenizing: {i} of {len(corpus)}", end="\r")
            detokenized_corpus += self.vocab_list[corpus[i]]
        if debug:
            print()
        return detokenized_corpus
