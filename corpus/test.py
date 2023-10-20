import os
import pandas as pd
import torch
from model import Model
from tokenizers.tokenizer import Tokenizer
from typing import List
from torch.utils.data import Dataset
import torch
from utils import SubstringDataset, print_parameters
from embedding import Embedding, UnEmbedding
from tqdm import tqdm
import sys

DIR = "corpus/refinedweb/falcon-refinedweb/data/"
# iterate over all files in dir

# print total filesize in human-readable format
total_size = 0
for FILE in os.listdir(DIR):
    # print filesize
    print(FILE + " " + str(os.path.getsize(DIR + FILE)))
    total_size += os.path.getsize(DIR + FILE)
# in GB
print("Total size: " + str(total_size / 1024 / 1024 / 1024) + " GB")

# let FILE be the first file.
FILE = os.listdir(DIR)[0]

df = pd.read_parquet(DIR + FILE, engine='fastparquet')
print(df.shape)
# print the first row
print(df.iloc[1])

# print the "content" of the first row
print(df.iloc[1]["content"])
train_corpus = bytes(df.iloc[0]["content"], encoding="utf-8")
train_corpus = "".join([chr(i) for i in train_corpus])

TOKENIZER_SOURCE = "./tokenizers/tokenizer_outputs/mahabharata_size4000_cap10.txt"
tokenizer = Tokenizer()
tokenizer.load(TOKENIZER_SOURCE)


tokenized_train_corpus: torch.Tensor = torch.tensor(
    tokenizer.tokenize(train_corpus), dtype=torch.int64)
# print size of tokenized_train_corpus
print(tokenized_train_corpus.size())


# for FILE in os.listdir(DIR):
#     # print filesize
#     print(FILE + " " + str(os.path.getsize(DIR + FILE)))
#     # check whether file is in parquet format
#     if FILE.endswith(".parquet"):
#         # read the file
#         df = pd.read_parquet(DIR + FILE, engine='fastparquet')
#         print(df.shape)

# # print total filesize in human-readable format
# print("Total size: " + str(os.path.getsize(DIR)) + " bytes")
