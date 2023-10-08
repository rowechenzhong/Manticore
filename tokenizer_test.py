from tokenizer import BPE

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

test = BPE(vocab_size=500)

corpus1 = open("mahabharata1.txt").read()
corpus2 = open("mahabharata2.txt").read()
corpus3 = open("mahabharata3.txt").read()
corpus = corpus1 + corpus2 + corpus3
print(corpus[:100])

int_corpus = test.to_ints(corpus)
test.train(int_corpus)


# print only printable characters
print([i for i in test.vocab if i.isprintable()])

tokenized_corpus = test.tokenize(corpus)

print(tokenized_corpus[:100])

detokenized_corpus = test.detokenize(tokenized_corpus)

print("Compression ratio is ", len(tokenized_corpus) / len(corpus))

print(detokenized_corpus[:100])
print(detokenized_corpus == corpus)
