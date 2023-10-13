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

    def __init__(self, model: Model, tokenizer: Tokenizer, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def train(self, train_corpus: str,
              test_corpus: str,
              size: int = 100,
              batch_size: int = 1000,
              epochs: int = 10) -> None:
        """
        Train the model on a corpus, which is just a list.
        The model will be trained to predict the next token,
        and the loss will be cross-entropy.
        """
        # tokenized_train_corpus = torch.tensor(
        #     self.tokenizer.tokenize(train_corpus))
        # use typing
        tokenized_train_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(train_corpus), dtype=torch.int64)

        train = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_train_corpus, size), batch_size=batch_size, shuffle=True).to(self.device)

        tokenized_test_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(test_corpus), dtype=torch.int64)
        test = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_test_corpus, size), batch_size=batch_size, shuffle=True).to(self.device)

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
            for x, y in train:
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
            with torch.no_grad():
                for x, y in test:
                    y_pred = self.model(x).permute(0, 2, 1)
                    l = loss(y_pred, y)
                    test_loss += l.item()
            print(f"Test loss: {test_loss / len(test)}")

    def generate(self, seed: str, length: int = 100) -> str:
        """
        Generate a string of a given length from a given seed.
        """
        tokenized_seed: List[int] = torch.tensor(
            self.tokenizer.tokenize(seed))
        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                y_pred = self.model(tokenized_seed[-100:])
                tokenized_seed = torch.cat(
                    (tokenized_seed, torch.argmax(y_pred, dim=1)[-1].unsqueeze(0)))
        return self.tokenizer.detokenize(tokenized_seed.tolist())

    def save(self, path: str) -> None:
        """
        Save the model and tokenizer to a given path.
        """
        torch.save(self.model.state_dict(), path + ".model")
        self.tokenizer.save(path + ".tokenizer")

    def load(self, path: str) -> None:
        """
        Load the model and tokenizer from a given path.
        """
        self.model.load_state_dict(torch.load(path + ".model"))
        self.tokenizer.load(path + ".tokenizer")


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.load(
        "./tokenizers/tokenizer_outputs/mahabharata_size4000_cap10.txt")

    corpus1 = open("corpus\\mahabharata1.txt", "rb").read()
    corpus1 = "".join([chr(i) for i in corpus1])
    corpus2 = open("corpus\\mahabharata2.txt", "rb").read()
    corpus2 = "".join([chr(i) for i in corpus2])
    corpus3 = open("corpus\\mahabharata3.txt", "rb").read()
    corpus3 = "".join([chr(i) for i in corpus3])
    corpus = corpus1 + corpus2 + corpus3

    train_corpus = corpus1 + corpus2
    test_corpus = corpus3

    embedding_in = Embedding(len(tokenizer), 64)
    embedding_out = UnEmbedding(len(tokenizer), 64)

    model = Model(embedding_in, embedding_out, size=64, layers=8)
    manticore = Manticore(model, tokenizer)

    manticore.train(train_corpus, test_corpus)
    # prayge
    print(manticore.generate("The ", 100))

    manticore.save("manticore")
    manticore.load("manticore")
    print(manticore.generate("The ", 100))
