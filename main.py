from model import Model
from tokenizers.tokenizer import Tokenizer
import torch
from utils import print_parameters
from embedding import Embedding, UnEmbedding
from manticore import Manticore
# Path: manticore.py

if __name__ == "__main__":
    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    EXPERIMENT = 2
    LOAD_SAVED = True
    LOAD_EPOCH = 4
    DEBUG = False

    if not LOAD_SAVED:
        LOAD_EPOCH = -1

    if EXPERIMENT == 1:
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

    transformer_params = {"size": SIZE, "size_internal": SIZE *
                          4, "attention_size": SIZE * 4, "decoder": True}
    tokenizer = Tokenizer()
    tokenizer.load(TOKENIZER_SOURCE)

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

    manticore.train(train_corpus, test_corpus, **train_kwargs)

    manticore.save(SAVE_NAME)

    print(
        manticore.generate(
            r"""II.  It is high time that Communists should openly, in the
face of the whole world, publish their views, """, 1000
        )
    )
