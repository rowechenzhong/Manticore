"""
Read from Falcon RefinedWeb dataset, and dump it into chunks into the desired path.
"""

import argparse
from datasets import load_dataset
from tqdm import tqdm

DATASET_CHUNK_SIZE = 1024
CORPUS_DIR = "corpus/falcon/"
FALCON_DATASET = "tiiuae/falcon-refinedweb"
DELIMITER = chr(30)
SHUFFLE_SEED = 42


def get_content(chunk_number: int, split: str, content): 
    """
    Save the results of exactly one dataset and move the stream forward
    """
    # Get the content and skip forward
    to_write = DELIMITER.join([row['content'] for row in content])

    # Save the content
    with open(CORPUS_DIR + split + "/" + str(chunk_number) + ".txt", "w", encoding='utf-8') as f:
        f.write(to_write)
    

def stream_content(start_chunk: int, end_chunk: int, split: str = 'train'):
    """
    Stream in the content of many pieces of the dataset into the directory.
    """

    if start_chunk is None or end_chunk is None:
        raise ValueError("Must specify start and end chunk")

    # Load and skip the dataset
    print("Loading dataset...")
    stream = load_dataset(FALCON_DATASET, split=split, streaming=True)
    print("Done!")
    # Very important! Shuffle the dataset
    print("Shuffling and skipping...")
    stream = stream.shuffle(seed=SHUFFLE_SEED)
    stream = stream.skip(start_chunk * DATASET_CHUNK_SIZE)
    print("Done!")

    for chunk_number in tqdm(range(start_chunk, end_chunk)):
        get_content(chunk_number, split, stream.take(DATASET_CHUNK_SIZE))
        stream = stream.skip(DATASET_CHUNK_SIZE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manticore Data Streamer')
    parser.add_argument('--train', action='store_true', help='Get train data')
    parser.add_argument('--test', action='store_true', help='Get test data')
    parser.add_argument('--ss', type=int, help='First chunk of the dataset')
    parser.add_argument('--to', type=int, help='Last chunk of the dataset')
    args = parser.parse_args()

    if args.train:
        stream_content(start_chunk=args.ss, end_chunk=args.to, split='train')
    elif args.test:
        stream_content(start_chunk=args.ss, end_chunk=args.to, split = 'test')
    else:
        parser.print_help()