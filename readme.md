Test 2:
Parameters:

```
corpus1 = open("./corpus/mahabharata1.txt", "rb").read()
corpus1 = "".join([chr(i) for i in corpus1])
corpus2 = open("./corpus/mahabharata2.txt", "rb").read()
corpus2 = "".join([chr(i) for i in corpus2])
corpus3 = open("./corpus/mahabharata3.txt", "rb").read()
corpus3 = "".join([chr(i) for i in corpus3])

TOKENIZER_SOURCE = "./tokenizers/tokenizer_outputs/mahabharata_size4000_cap10.txt"
train_corpus = corpus1 + corpus2
test_corpus = corpus3

SIZE = 256
SEQ_LEN = 120
LAYERS = 2
BATCH_SIZE = 1000
EPOCHS = 10
DEBUG = False
SAVE_PER_EPOCH = True

SAVE_NAME = "mahabharata_manticore_smol"
```

Results:

```
Epoch 0:
Training loss: 2.1934340821049076
Test loss: 2.1952488116632756
```
