from model import Model
from tokenizers.tokenizer import Tokenizer
import torch
from utils import SubstringDataset
from tqdm import tqdm
import sys

# Path: manticore.py


class Manticore:
    """
    The Manticore class is a wrapper for the model and tokenizer.
    """

    def __init__(self, model: Model, tokenizer: Tokenizer, device: str = "cpu") -> None:
        self.model: Model = model.to(device)
        self.tokenizer: Tokenizer = tokenizer
        self.device: str = device

    def train(self,
              train_corpus: str,
              test_corpus: str,

              seq_len: int = 100,
              batch_size: int = 100,

              dirty: float = 0.1,

              epochs: int = 10,
              start_epoch: int = 0,

              debug: bool = False,
              save_per_epoch: bool = False,
              save_name: str = False) -> None:
        """
        Train the model on a corpus, which is just a list.
        The model will be trained to predict the next token,
        and the loss will be cross-entropy.

        :param train_corpus: The corpus to train on.
        :param test_corpus: The corpus to test on.

        :param seq_len: The length of the sequences to train on.
        :param batch_size: The batch size to use.

        :param dirty: The percentage of input to corrupt.

        :param epochs: The number of epochs to train for.
        :param start_epoch: The epoch to start at.

        :param debug: Whether to print debug information.
        :param save_per_epoch: Whether to save the model after each epoch.
        :param save_name: The name to save the model as.
        """
        tokenized_train_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(train_corpus), dtype=torch.int64)

        train = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_train_corpus, seq_len), batch_size=batch_size, shuffle=True, pin_memory=True)

        tokenized_test_corpus: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(test_corpus), dtype=torch.int64)
        test = torch.utils.data.DataLoader(
            SubstringDataset(tokenized_test_corpus, seq_len), batch_size=batch_size, shuffle=True, pin_memory=True)

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
        for epoch in range(start_epoch, epochs):
            print(f"Epoch {epoch}")
            train_loss = 0
            self.model.train()
            with tqdm(enumerate(train), total=len(train)) as pbar:
                for i, (x, y) in pbar:
                    training_loss += self.single_train_step(
                        x, y, loss, optimizer, dirty)
                    # write training loss to tqdm
                    pbar.set_postfix(
                        {"Train loss": train_loss / (i + 1)})

            print(f"Train loss: {train_loss / len(train)}")

            if save_per_epoch:
                self.save(save_name + "_epoch" + str(epoch))

            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                with tqdm(enumerate(test), total=len(test)) as pbar:
                    # for x, y in tqdm(test):
                    for i, (x, y) in pbar:
                        test_loss += self.single_test_step(x, y, loss)
                        pbar.set_postfix({"Test loss": test_loss / (i + 1)})
                    if debug:
                        self.single_test_step(x, y, loss, debug=True)
            print(f"Test loss: {test_loss / len(test)}")

    def single_train_step(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          loss: torch.nn.NLLLoss,
                          optimizer: torch.optim.Adam,
                          dirty: float) -> None:
        """
        Train the model on a single batch.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        optimizer.zero_grad()

        # corrupt input
        random_indices = torch.rand(x.shape) < dirty
        random_tokens = torch.randint(
            0, len(self.tokenizer), x.shape, dtype=torch.int64)

        x[random_indices] = random_tokens[random_indices]

        # print("In train loop")
        # print(x.shape, y.shape)
        # print(x.dtype, y.dtype)
        y_pred = self.model(x).permute(0, 2, 1)

        # only incur loss on last half of the sequence
        # y_pred = y_pred[:, :, -seq_len//2:]
        # y = y[:, -seq_len//2:]

        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        return l.item()

    def single_test_step(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         loss: torch.nn.NLLLoss,
                         debug: bool = False) -> None:
        """
        Test the model on a single batch.
        """
        x = x.to(self.device)  # (batch_size, seq_len)
        y = y.to(self.device)  # (batch_size, seq_len)
        # (batch_size, vocab_size, seq_len)
        y_pred = self.model(x).permute(0, 2, 1)
        # only incur loss on last half of the sequence
        # y_pred = y_pred[:, :, -seq_len//2:]
        # y = y[:, -seq_len//2:]
        if debug:
            print("In test loop")
            print("x:", x.shape, x.dtype)
            print(self.tokenizer.detokenize(x[0].tolist()))  # (seq_len)
            print("y:", y.shape, y.dtype)
            print(self.tokenizer.detokenize(y[0].tolist()))  # (seq_len)
            # report top 5 predictions
            print("y_pred:", y_pred.shape, y_pred.dtype)
            top_5 = torch.topk(y_pred[0], 5, dim=0)  # (5, seq_len)
            for i in range(5):
                print(self.tokenizer.detokenize(top_5.indices[i].tolist()))
                # print(top_5.values[i])
        return loss(y_pred, y).item()

    def generate(self, seed: str, length: int = 100) -> str:
        """
        Generate a string of a given length from a given seed.
        """
        tokenized_seed: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(seed)).unsqueeze(0).to(self.device)  # (1, seq_len)

        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                # Forward pass!
                y_pred = self.model(tokenized_seed)

                # Sample probability distribution
                result = torch.multinomial(
                    torch.exp(y_pred[:, -1, :].squeeze(1)), 1).to(self.device)

                tokenized_seed = torch.cat(
                    (tokenized_seed, result), dim=1).to(self.device)

                print(self.tokenizer.vocab_list[tokenized_seed[0, -1]], end='')
                sys.stdout.flush()
        tokenized_seed = tokenized_seed.squeeze(0).tolist()
        return self.tokenizer.detokenize(tokenized_seed)

    def save(self, path: str) -> None:
        """
        Save the model to a given path.
        """
        torch.save(self.model.state_dict(), path + ".model")

    def load(self, path: str) -> None:
        """
        Load the model from a given path.
        """
        # Be sure to load the model to cpu first.
        self.model.load_state_dict(torch.load(
            path + ".model", map_location=torch.device("cpu")))
        self.model.to(self.device)

    def generate_beamsearch_once(self, tokenized_seed: torch.Tensor, breadth: int, depth: int) -> torch.Tensor:
        """
        Beam search helper. Generates one token with parameters as described below.
        self.model should be in eval() mode, and this should be in a torch.no_grad() environment.
        """
        # Keeping track of the top results
        top = [(torch.tensor(1.0, dtype=torch.float64),
                torch.tensor([[]], dtype=torch.int64))]
        for _ in range(depth):
            nxtop = list()
            for pr, tks in top:
                # Generate this branch's tokens
                cur_seed = torch.cat((tokenized_seed, tks),
                                     dim=1).to(self.device)

                # Forward pass!
                y_pred = self.model(cur_seed)

                # Sample probability distribution
                probs = torch.exp(y_pred[:, -1, :].squeeze(1))

                # Take the top few indices
                best = torch.topk(probs, breadth)

                # Add the new paths
                for idx in range(breadth):
                    nxtop.append((pr * best.values[0, idx], torch.cat(
                        (tks, best.indices[:, idx:idx+1]), dim=1).to(self.device)))

            # Filter top array
            top = sorted(nxtop, key=lambda item: item[0].item(), reverse=True)[
                :breadth]

        # Get the best path
        print(self.tokenizer.detokenize(top[0][1].squeeze(0).tolist()), end='')
        sys.stdout.flush()
        return torch.cat((tokenized_seed, top[0][1]), dim=1).to(self.device)

    def generate_beamsearch(self, seed: str, length: int = 30, breadth: int = 50, depth: int = 10) -> str:
        """
        Use Beam Search to generate a string of a given length from a given seed.
        - breadth: width of beam search to use
        - depth: how deep to search
        """
        tokenized_seed: torch.Tensor = torch.tensor(
            self.tokenizer.tokenize(seed)).unsqueeze(0).to(self.device)  # (1, seq_len)

        self.model.eval()
        with torch.no_grad():
            for _ in range(length):
                tokenized_seed = self.generate_beamsearch_once(
                    tokenized_seed, breadth, depth)
                tokenized_seed = tokenized_seed[:, -100:]

        tokenized_seed = tokenized_seed.squeeze(0).tolist()
        return self.tokenizer.detokenize(tokenized_seed)

    def chat_endlessly(self) -> None:
        """
        Chatbot mode!
        """
        while True:
            print("Your turn! > ", end='')
            prompt = input()
            print()
            print("Manticore says > ", end='')
            self.generate_beamsearch(prompt, 30)
            print()
