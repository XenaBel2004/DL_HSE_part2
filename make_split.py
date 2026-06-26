import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet("data/vk_working_subset.parquet")

y = df["like"].astype(int).values
idx = np.arange(len(df))

train_idx, test_idx = train_test_split(
    idx,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

np.save("data/train_idx.npy", train_idx)
np.save("data/test_idx.npy", test_idx)

print("train:", len(train_idx))
print("test:", len(test_idx))