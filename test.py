import pandas as pd

# read FILE
FILE = "corpus/train-00000-of-05534-b8fc5348cbe605a5.parquet"
# read the file
df = pd.read_parquet(FILE, engine='pyarrow')
# print the first 5 rows
print(df.head())
# print the shape of the dataframe
print(df.shape)
# print the columns of the dataframe
print(df.columns)
