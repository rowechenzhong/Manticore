from model import Model
from tokenizers.BPE_with_Pretokenization.aux_python_tokenizer import BPE
from typing import List
from torch.utils.data import Dataset
import torch
# Path: manticore.py


class SubstringDataset(Dataset):
    """
    This class is a PyTorch Dataset that returns substrings of a corpus.
    """

    def __init__(self, corpus: torch.Tensor, size: int = 100) -> None:
        """
        Initialize the dataset.
        """
        self.corpus = corpus
        self.size = size

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.corpus) - self.size

    def __getitem__(self, i: int) -> str:
        """
        Return the substring of the corpus at index i.
        """
        return self.corpus[i:i + self.size], self.corpus[i + 1:i + self.size + 1]


class Manticore:
    """
    This class comprises a model and tokenizer, 
    """

    def __init__(self, model: Model, tokenizer: BPE):
        self.model = model
        self.tokenizer = tokenizer

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
        tokenized_train_corpus: List[int] = torch.tensor(
            self.tokenizer.tokenize(train_corpus))
        train = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_train_corpus, size), batch_size=batch_size)

        tokenized_test_corpus: List[int] = torch.tensor(
            self.tokenizer.tokenize(test_corpus))
        test = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_test_corpus, size), batch_size=batch_size)

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            trainings_loss = 0
            self.model.train()
            for x, y in train:
                optimizer.zero_grad()
                y_pred = self.model(x)
                l = loss(y_pred, y)
                trainings_loss += l.item()
                l.backward()
                optimizer.step()
            print(f"Training loss: {trainings_loss / len(train)}")
            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for x, y in test:
                    y_pred = self.model(x)
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
    # test = BPE(vocab_size=1024)
    # corpus = open("atoms.txt").read()
    # int_corpus = test.to_ints(corpus)
    # test.train(int_corpus)
    # tokenized_corpus = test.tokenize(corpus)
    # detokenized_corpus = test.detokenize(tokenized_corpus)
    # print("Compression ratio is ", len(tokenized_corpus) / len(corpus))
    # print(detokenized_corpus[:100])
    # print(detokenized_corpus == corpus)
    test = BPE(vocab_size=500)
    corpus1 = open("mahabharata1.txt").read()
    corpus2 = open("mahabharata2.txt").read()
    corpus3 = open("mahabharata3.txt").read()
    corpus = corpus1 + corpus2 + corpus3
    int_corpus = test.to_ints(corpus)
    test.train(int_corpus)
    tokenized_corpus = test.tokenize(corpus)
    detokenized_corpus = test.detokenize(tokenized_corpus)
    print("Compression ratio is ", len(tokenized_corpus) / len(corpus))
    print(detokenized_corpus[:100])
    print(detokenized_corpus == corpus)
    model = Model(500, 100, 100)
    manticore = Manticore(model, test)
    manticore.train(corpus, corpus)
    print(manticore.generate("The ", 100))
    manticore.save("manticore")
    manticore.load("manticore")
    print(manticore.generate("The ", 100))
