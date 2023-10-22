from model import Model
from tokenizers.tokenizer import Tokenizer
import torch
from utils import print_parameters
from embedding import Embedding, UnEmbedding
from manticore import Manticore
from falcon_stream import FalconStreamer, BatchStreamer
# Path: manticore.py

if __name__ == "__main__":
    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    EXPERIMENT = 5
    LOAD_SAVED = False
    LOAD_EPOCH = 0
    DEBUG = False

    if not LOAD_SAVED:
        LOAD_EPOCH = -1

    if EXPERIMENT == 1:
        raise NotImplementedError
        corpus = open("./corpus/communistmanifesto.txt", "rb").read()
        corpus = "".join([chr(i) for i in corpus])
        train_corpus = corpus[:int(0.8 * len(corpus))]
        test_corpus = corpus[int(0.8 * len(corpus)):]

        TOKENIZER_SOURCE = "./tokenizers/tokenizer_outputs/mahabharata_size4000_cap10.txt"

        SIZE = 256
        LAYERS = 2
        SAVE_NAME = "communist_manticore_smol"

        train_kwargs = {
            "seq_len": 120,
            "batch_size": 100,
            "epochs": 30,
            "debug": DEBUG,
            "save_per_epoch": False,
            "save_name": "communist_manticore_smol",
            "start_epoch": LOAD_EPOCH + 1
        }

    elif EXPERIMENT == 2:
        raise NotImplementedError
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
        LAYERS = 2
        SAVE_NAME = "mahabharata_manticore_smol"

        train_kwargs = {
            "seq_len": 256,
            "batch_size": 1000,
            "epochs": 10,
            "debug": DEBUG,
            "save_per_epoch": True,
            "save_name": "mahabharata_manticore_smol",
            "start_epoch": LOAD_EPOCH + 1
        }

    elif EXPERIMENT == 3:
        raise NotImplementedError
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
        LAYERS = 80
        SAVE_NAME = "mahabharata_manticore_chungus"

        train_kwargs = {
            "seq_len": 256,
            "batch_size": 1000,
            "epochs": 10,
            "debug": DEBUG,
            "save_per_epoch": True,
            "save_name": "mahabharata_manticore_chungus",
            "start_epoch": LOAD_EPOCH + 1
        }

    elif EXPERIMENT == 4:
        train_corpus = FalconStreamer(mode="stream", split="train")
        test_corpus = FalconStreamer(mode="stream", split="test")

        SEQ_LEN = 256
        BATCH_SIZE = 1000
        EPOCH_SIZE = 100

        TOKENIZER_SOURCE = "./tokenizers/tokenizer_outputs/peregrine.txt"

        SIZE = 256
        LAYERS = 80
        SAVE_NAME = "falcon_manticore_chungus"

        train_kwargs = {
            "epochs": 10,
            "debug": DEBUG,
            "save_per_epoch": True,
            "save_name": SAVE_NAME,
            "start_epoch": LOAD_EPOCH + 1
        }

    elif EXPERIMENT == 5:
        train_corpus = FalconStreamer(mode="stream", split="train")
        test_corpus = FalconStreamer(mode="stream", split="test")
        SEQ_LEN = 128
        BATCH_SIZE = 10
        EPOCH_SIZE = 10

        TOKENIZER_SOURCE = "./tokenizers/tokenizer_outputs/peregrine.txt"

        SIZE = 32
        LAYERS = 2
        SAVE_NAME = "falcon_manticore_smol"

        # :param dirty: The percentage of input to corrupt.

        # :param epochs: The number of epochs to train for.
        # :param start_epoch: The epoch to start at.

        # :param debug: Whether to print debug information.
        # :param save_per_epoch: Whether to save the model after each epoch.
        # :param save_name: The name to save the model as.
        train_kwargs = {
            "epochs": 10,
            "debug": DEBUG,
            "save_per_epoch": True,
            "save_name": SAVE_NAME,
            "start_epoch": LOAD_EPOCH + 1
        }

    transformer_params = {"size": SIZE, "attention_size": SIZE * 4,
                          "heads": 8, "size_internal": SIZE *
                          4, "decoder": True}

    tokenizer = Tokenizer()

    # Delimiter is character 31
    tokenizer.load(TOKENIZER_SOURCE, delimiter=chr(31))

    # Num is the number of batches to train on per epoch. This is tunable.
    train_streamer = BatchStreamer(
        train_corpus, tokenizer, batch_size=BATCH_SIZE,
        context_length=SEQ_LEN, num=10
    )

    test_streamer = BatchStreamer(
        test_corpus, tokenizer, batch_size=BATCH_SIZE,
        context_length=SEQ_LEN, num=10
    )

    embedding_in = Embedding(len(tokenizer), SIZE)
    embedding_out = UnEmbedding(len(tokenizer), SIZE)

    model = Model(embedding_in, embedding_out,
                  transformer_params, layers=LAYERS)

    print_parameters(model, DEBUG=DEBUG)

    manticore = Manticore(model, tokenizer, device=device)

    if LOAD_SAVED:
        manticore.load(SAVE_NAME + "_epoch" + str(LOAD_EPOCH))
        manticore.chat_endlessly()
        print(manticore.generate(
            """"The holy one said, 'O thou of great wisdom, I desire to hear in detail, O thou that art conversant with the 
duties of the science of Profit, and thou art the foremost of all wielders of """,
            100)
        )

    manticore.stream_train(train_streamer, test_streamer, **train_kwargs)

    manticore.save(SAVE_NAME)

    print(
        manticore.generate(
            r"""II.  It is high time that Communists should openly, in the
face of the whole world, publish their views, """, 1000
        )
    )
