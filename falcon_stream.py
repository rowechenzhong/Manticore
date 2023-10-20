"""
Read from Falcon RefinedWeb dataset, and dump it into chunks into the desired path.
"""

import argparse
from datasets import load_dataset
from tokenizers import tokenizer
from tqdm import tqdm

DATASET_CHUNK_SIZE = 1024
CORPUS_DIR = "corpus/falcon/"
FALCON_DATASET = "tiiuae/falcon-refinedweb"
DELIMITER = chr(30)
SHUFFLE_SEED = 42
TEST_FREQUENCY = 5


class FalconStreamer:
    def __init__(self, mode='stream', split='train'):
        """
        Initialize the boi
        """
        if mode not in ['stream', 'write']:
            raise ValueError("Mode must be 'stream' or 'write'")

        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'")

        self.mode = mode
        self.split = split

        # Load and skip the dataset
        print("Loading dataset...")

        self.dataset = load_dataset(
            FALCON_DATASET, split='train', streaming=True)

        # Very important! Shuffle the dataset
        print("Shuffling...")
        self.dataset = self.dataset.shuffle(seed=SHUFFLE_SEED)

    def __iter__(self):
        """
        Prepare for iteration
        """
        assert self.mode == 'stream', "Must be 'stream' mode to stream content"
        self.cur_streaming_row = 0
        self.stream = iter(self.dataset)
        return self

    def __next__(self):
        """
        Get the next row's content
        """
        assert self.mode == 'stream', "Must be 'stream' mode to stream content"
        try:
            # Ah yes.
            while self.split == 'train' and (self.cur_streaming_row // DATASET_CHUNK_SIZE) % TEST_FREQUENCY == 0 or self.split == 'test' and (self.cur_streaming_row // DATASET_CHUNK_SIZE) % TEST_FREQUENCY != 0:
                self.cur_streaming_row += 1
                try:
                    next(self.stream)
                except StopIteration:
                    raise StopIteration
            return next(self.stream)['content']
        except StopIteration:
            raise StopIteration

    def get_content(self, chunk_number: int, stream, skip: bool = False):
        """
        Save the results of exactly one dataset and move the stream forward
        """
        assert self.mode == 'write', "Must be 'write' mode to write content"

        to_write = []
        for _ in range(DATASET_CHUNK_SIZE):
            try:
                to_write.append(next(stream)['content'])
            except StopIteration:
                break

        if skip:
            return

        to_write = DELIMITER.join(to_write)
        if to_write[-1] != DELIMITER:
            to_write += DELIMITER

        # Save the content
        with open(CORPUS_DIR + self.split + "/" + str(chunk_number) + ".txt", "w", encoding='utf-8') as f:
            f.write(to_write)

    def stream_content(self, start_chunk: int, end_chunk: int):
        """
        Stream in the content of many pieces of the dataset into the directory.
        """
        assert self.mode == 'write', "Must be 'write' mode to write content"

        if start_chunk is None or end_chunk is None:
            raise ValueError("Must specify start and end chunk")

        self.stream = iter(self.dataset.skip(start_chunk * DATASET_CHUNK_SIZE))

        for chunk_number in tqdm(range(start_chunk, end_chunk)):
            if self.split == 'train':
                if chunk_number % TEST_FREQUENCY != 0:
                    self.get_content(chunk_number, self.stream)
                else:
                    self.get_content(chunk_number, self.stream, skip=True)
            elif self.split == 'test':
                if chunk_number % TEST_FREQUENCY == 0:
                    self.get_content(chunk_number, self.stream)
                else:
                    self.get_content(chunk_number, self.stream, skip=True)


class BatchStreamer:
    def __init__(self, stream: FalconStreamer, tokenizer: tokenizer, batch_size: int = 64):
        """
        Initialize the boi
        """
        self.stream = stream
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        """
        Grab the next batch of batch_size rows, pad and tokenize them all
        """
        # for rowechen, because i am lost


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manticore Data Streamer')
    parser.add_argument('--train', action='store_true', help='Get train data')
    parser.add_argument('--test', action='store_true', help='Get test data')
    parser.add_argument('--ss', type=int, help='First chunk of the dataset')
    parser.add_argument('--to', type=int, help='Last chunk of the dataset')
    args = parser.parse_args()

    if args.train:
        FalconStreamer(mode='write', split='train').stream_content(
            start_chunk=args.ss, end_chunk=args.to)
    elif args.test:
        FalconStreamer(mode='write', split='test').stream_content(
            start_chunk=args.ss, end_chunk=args.to)
    else:
        parser.print_help()
