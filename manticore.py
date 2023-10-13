from model import Model
from tokenizers.tokenizer import Tokenizer
from typing import List
from torch.utils.data import Dataset
import torch
from utils import SubstringDataset
from embedding import Embedding, UnEmbedding
from tqdm import tqdm
# Path: manticore.py


class Manticore:
    """
    This class comprises a model and tokenizer, 
    """

    def __init__(self, model: Model, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, train_corpus: str,
              test_corpus: str,
              size: int = 100,
              batch_size: int = 100,
              epochs: int = 10,
              debug: bool = False) -> None:
        """
        Train the model on a corpus, which is just a list.
        The model will be trained to predict the next token,
        and the loss will be cross-entropy.

        :param train_corpus: The corpus to train on.
        :param test_corpus: The corpus to test on.
        :param size: The length of a training example. (Called seq_len elsewhere)
        :param batch_size: The batch size.
        :param epochs: The number of epochs to train for.
        """
        # tokenized_train_corpus = torch.tensor(
        #     self.tokenizer.tokenize(train_corpus))
        # use typing
        tokenized_train_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(train_corpus), dtype=torch.int64)

        train = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_train_corpus, size), batch_size=batch_size, shuffle=True, pin_memory=True)

        tokenized_test_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(test_corpus), dtype=torch.int64)
        test = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_test_corpus, size), batch_size=batch_size, shuffle=True, pin_memory=True)

        # The purpose of pin_memory is to speed up the transfer of data from CPU to GPU, which is done by transferring
        # the data to the pinned memory first and then transferring it to the GPU. This is because the data is transferred
        # from the CPU to the GPU through the PCI-E bus. The PCI-E bus is shared by all devices on the motherboard,
        # and the data transfer is slow. The pinned memory is allocated in the memory area that is not shared by the
        # PCI-E bus, so the data transfer is fast. The disadvantage is that the pinned memory cannot be
        # shared by multiple processes, so it is not suitable for multi-process data reading.

        print("Size of training set:", len(train))
        print("Size of test set:", len(test))
        # note that tokenizer already outputs log-probabilities,
        # so we don't need to apply softmax. Just use a NLLL loss.
        loss = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            trainings_loss = 0
            self.model.train()
            # tqdm with time estimate
            for x, y in tqdm(train):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                # print("In train loop")
                # print(x.shape, y.shape)
                # print(x.dtype, y.dtype)
                y_pred = self.model(x).permute(0, 2, 1)
                l = loss(y_pred, y)
                trainings_loss += l.item()
                l.backward()
                optimizer.step()
            print(f"Training loss: {trainings_loss / len(train)}")
            test_loss = 0
            self.model.eval()
            one_shown = False
            with torch.no_grad():
                for x, y in test:
                    x = x.to(self.device)  # (batch_size, seq_len)
                    y = y.to(self.device)  # (batch_size, seq_len)
                    # (batch_size, vocab_size, seq_len)
                    y_pred = self.model(x).permute(0, 2, 1)
                    l = loss(y_pred, y)
                    test_loss += l.item()

                    if debug and not one_shown:
                        print("In test loop")
                        print("x:", x.shape, x.dtype)
                        print(self.tokenizer.detokenize(
                            x[0].tolist()))  # (seq_len)
                        print("y:", y.shape, y.dtype)
                        print(self.tokenizer.detokenize(
                            y[0].tolist()))  # (seq_len)
                        # report top 5 predictions
                        print("y_pred:", y_pred.shape, y_pred.dtype)
                        top_5 = torch.topk(y_pred[0], 5, dim=0)  # (5, seq_len)
                        for i in range(5):
                            print(self.tokenizer.detokenize(
                                top_5.indices[i].tolist()))
                            print(top_5.values[i])
                        one_shown = True
            print(f"Test loss: {test_loss / len(test)}")

    def generate(self, seed: str, length: int = 100) -> str:
        """
        Generate a string of a given length from a given seed.
        """
        tokenized_seed: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(seed)).unsqueeze(0)  # (1, seq_len)

        tokenized_seed.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                y_pred = self.model(tokenized_seed[-100:])
                print(y_pred.shape)
                # only take the last token
                y_pred = y_pred[:, -1, :].unsqueeze(1)
                result = torch.argmax(y_pred, dim=2)
                print(tokenized_seed.shape, result.shape)

                tokenized_seed = torch.cat((tokenized_seed, result), dim=1)
        tokenized_seed = tokenized_seed.squeeze(0).tolist()
        return self.tokenizer.detokenize(tokenized_seed)

    def save(self, path: str) -> None:
        """
        Save the model and tokenizer to a given path.
        """
        torch.save(self.model.state_dict(), path + ".model")

    def load(self, path: str) -> None:
        """
        Load the model and tokenizer from a given path.
        """
        self.model.load_state_dict(torch.load(path + ".model"))

########## Nerfed test ##########


if __name__ == "__main__":
    # detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    NERFED = False

    if NERFED:
        corpus = open("corpus\\communistmanifesto.txt", "rb").read()
        corpus = "".join([chr(i) for i in corpus])
        train_corpus = corpus[:int(0.8 * len(corpus))]
        test_corpus = corpus[int(0.8 * len(corpus)):]

        SIZE = 20
        LAYERS = 2
        BATCH_SIZE = 20
        EPOCHS = 2
        DEBUG = True
    else:
        corpus1 = open("corpus\\mahabharata1.txt", "rb").read()
        corpus1 = "".join([chr(i) for i in corpus1])
        corpus2 = open("corpus\\mahabharata2.txt", "rb").read()
        corpus2 = "".join([chr(i) for i in corpus2])
        corpus3 = open("corpus\\mahabharata3.txt", "rb").read()
        corpus3 = "".join([chr(i) for i in corpus3])

        train_corpus = corpus1 + corpus2
        test_corpus = corpus3

        SIZE = 64
        LAYERS = 8
        BATCH_SIZE = 100
        EPOCHS = 10
        DEBUG = False

    tokenizer = Tokenizer()
    tokenizer.load(
        "./tokenizers/tokenizer_outputs/mahabharata_size4000_cap10.txt")

    embedding_in = Embedding(len(tokenizer), SIZE)
    embedding_out = UnEmbedding(len(tokenizer), SIZE)

    model = Model(embedding_in, embedding_out,
                  size=SIZE, layers=LAYERS, decoder=True)
    if DEBUG:
        print("Parameters:")
        for name, param in model.named_parameters():
            print(name, param.shape)
    print("Total parameters:", sum([param.numel()
          for param in model.parameters()]))
    manticore = Manticore(model, tokenizer, device=device)

    manticore.train(train_corpus, test_corpus,
                    size=SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS, debug=DEBUG)

    manticore.save("manticore_chungus")
    print(manticore.generate("The ", 100))

    manticore.load("manticore_chungus")
    # prayge

    print(manticore.generate("The ", 100))
