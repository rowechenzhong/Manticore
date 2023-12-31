import os
from tokenizer import Tokenizer
# # Atoms test

# test = BPE(vocab_size=1024)

# corpus = open("atoms.txt").read()
# print(corpus[:100])

# int_corpus = test.to_ints(corpus)
# test.train(int_corpus)


# # print only printable characters
# print([i for i in test.vocab if i.isprintable()])

# tokenized_corpus = test.tokenize(corpus)

# print(tokenized_corpus[:100])

# detokenized_corpus = test.detokenize(tokenized_corpus)

# print("Compression ratio is ", len(tokenized_corpus) / len(corpus))

# print(detokenized_corpus[:100])
# print(detokenized_corpus == corpus)


# Mahabharata test

test: Tokenizer = Tokenizer()
# print current directory
# print(os.getcwd())
# You should always run python files from the root directory for consistency.

test.load("tokenizers\\tokenizer_outputs\\mahabharata_size4000_cap10.txt", debug=True)

corpus1 = open("corpus\\mahabharata1.txt", "rb").read()
corpus1 = "".join([chr(i) for i in corpus1])
corpus2 = open("corpus\\mahabharata2.txt", "rb").read()
corpus2 = "".join([chr(i) for i in corpus2])
corpus3 = open("corpus\\mahabharata3.txt", "rb").read()
corpus3 = "".join([chr(i) for i in corpus3])
corpus = corpus1 + corpus2 + corpus3
print(corpus[:100])

tokenized_corpus = test.tokenize(corpus, debug=True)

print(tokenized_corpus[:100])

detokenized_corpus = test.detokenize(tokenized_corpus)

print("Compression ratio is ", len(tokenized_corpus) / len(corpus))

print(detokenized_corpus[:100])
print(detokenized_corpus == corpus)
