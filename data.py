from datasets import load_dataset
import pandas as pd

ds = load_dataset("MLBtrio/genz-slang-dataset")
df = pd.DataFrame(ds['train'])

print(df.head())