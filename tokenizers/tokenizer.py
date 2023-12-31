from typing import List, Tuple, Dict, ByteString

MAX_TOKEN = 100
TOKEN_DELIMITER = chr(31)


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
    def __init__(self, EOS="<<EOS>>"):
        """
        Alright, let's create a BPE tokenizer.
        """
        self.vocab: TrieNode = TrieNode()
        # This is the trie. It does not contain the EOS token.
        self.vocab_list: List[str] = []
        # This is the list of tokens. The last one will be the EOS token.
        self.max_token_length = MAX_TOKEN  # This is the max token length
        self.EOS = EOS  # This is the end of sentence token
        self.EOS_idx = -1  # This is the index of the end of sentence token

    def __len__(self):
        return len(self.vocab_list)

    def __get_EOS__(self):
        return self.EOS

    def load(self, vocab_file, delimiter=TOKEN_DELIMITER, debug=False):
        self.vocab_list = []
        with open(vocab_file, "rb") as f:
            # read f as a list of bytes
            all_text = f.read()
            # convert bytes to string
            all_text = "".join([chr(i) for i in all_text])
            all_text = list(filter(lambda x: len(x) > 0,
                            all_text.split(delimiter)))

        self.vocab_list = all_text

        if debug:
            length_data = [0 for i in range(100)]
            for token in self.vocab_list:
                length_data[len(token)] += 1

            print("Loaded vocab of size ", len(self.vocab_list))
            for i in range(self.max_token_length + 1):
                print(i, length_data[i])

        self.vocab = TrieNode.create_vocab_trie(self.vocab_list)

        self.vocab_list.append(self.EOS)
        self.EOS_idx = len(self.vocab_list) - 1

    def tokenize(self, corpus: bytes, debug: bool = False, size=None) -> List[int]:
        """
        Tokenize the corpus.

        :param corpus: Input corpus as a string of bytes.
        :param debug: Print debug information.
        :param size: Size of the corpus to tokenize. If None, tokenize the entire corpus.
        If an integer, will tokenizer the first size characters of the corpus, then pad with EOS tokens.
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
                # Indexing into a bytes object returns an int!
                chr = corpus[rowechen_ptr]  # ord(corpus[rowechen_ptr])
            if chr not in cur.children:
                tokenized_corpus.append(isaac_notes)
                rowechen_ptr = isaac_ptr + 1
                if rowechen_ptr >= len(corpus):
                    break
                chr = corpus[rowechen_ptr]  # ord(corpus[rowechen_ptr])
                cur = self.vocab.children[chr][1]
            else:
                cur = cur.children[chr][1]

            if cur.mark != -1:
                isaac_ptr = rowechen_ptr
                isaac_notes = cur.mark

            rowechen_ptr += 1
            if rowechen_ptr & 0xfff == 0 and debug:
                print(
                    f"Tokenizing ........ {rowechen_ptr} of {len(corpus)}", end="\r")
        if debug:
            print("Finished tokenizing", end=" " * 29 + "\n")

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

        if size is not None:
            tokenized_corpus = tokenized_corpus[:size]
            tokenized_corpus += [self.EOS_idx] * \
                (size - len(tokenized_corpus))

        return tokenized_corpus

    def detokenize(self, corpus: List[int], debug: bool = False, EOS: bool = False) -> str:
        """
        Detokenize the corpus.

        :param corpus: Input corpus.
        :param debug: Print debug information.
        :param EOS: Whether to terminate the corpus on the first EOS token.
        """
        detokenized_corpus = ""
        for i in range(len(corpus)):
            if debug and i % 1000 == 0:
                print(f"Detokenizing: {i} of {len(corpus)}", end="\r")
            detokenized_corpus += self.vocab_list[corpus[i]]
            if self.vocab_list[corpus[i]] == self.EOS and EOS:
                break
        if debug:
            print()
        return detokenized_corpus
