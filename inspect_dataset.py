import pyarrow.parquet as pq
import pandas as pd
import numpy as np

pf = pq.ParquetFile("data/interactions/train/week_00.parquet")
batch = next(pf.iter_batches(batch_size=2_000_000))
inter = batch.to_pandas()

print("interactions:", inter.shape)

emb = np.load("data/metadata/item_embeddings.npz", allow_pickle=True)

item_ids = emb["item_id"]
item_embs = emb["embedding"]

emb_df = pd.DataFrame({
    "item_id": item_ids,
    "embedding": list(item_embs),
})

print("embeddings:", emb_df.shape)

inter_items = inter["item_id"].unique()
emb_df = emb_df[emb_df["item_id"].isin(inter_items)]

df = inter.merge(emb_df, on="item_id", how="inner")

print("merged:", df.shape)
print(df.columns.tolist())
print(df.head())

users = df["user_id"].drop_duplicates().sample(
    n=min(1000, df["user_id"].nunique()),
    random_state=42,
)

df = df[df["user_id"].isin(users)].copy()

print("subset:", df.shape)
print("users:", df["user_id"].nunique())
print("items:", df["item_id"].nunique())

# 6. сохраняем
df.to_parquet("data/vk_working_subset.parquet")

print("saved to data/vk_working_subset.parquet")

import pandas as pd

df = pd.read_parquet("data/vk_working_subset.parquet")

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(type(df["embedding"].iloc[0]))
print(len(df["embedding"].iloc[0]))