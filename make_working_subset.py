import pyarrow.parquet as pq
import pandas as pd
import numpy as np

INTERACTIONS_PATH = "data/interactions/train/week_00.parquet"
EMBEDDINGS_PATH = "data/metadata/item_embeddings.npz"
SAVE_PATH = "data/vk_working_subset.parquet"

# увеличивай постепенно
BATCH_SIZE = 100_000_000
N_USERS = 10_000
MIN_INTERACTIONS = 10

print("Reading interactions...")

pf = pq.ParquetFile(INTERACTIONS_PATH)
batch = next(pf.iter_batches(batch_size=BATCH_SIZE))
inter = batch.to_pandas()

print("interactions:", inter.shape)

print("Reading item embeddings...")

emb = np.load(EMBEDDINGS_PATH, allow_pickle=True)

emb_df = pd.DataFrame({
    "item_id": emb["item_id"],
    "embedding": list(emb["embedding"]),
})

print("embeddings:", emb_df.shape)

print("Merging...")

df = inter.merge(
    emb_df,
    on="item_id",
    how="inner",
)

print("merged:", df.shape)

print("Filtering active users...")

user_counts = df["user_id"].value_counts()

active_users = user_counts[
    user_counts >= MIN_INTERACTIONS
].index

df = df[df["user_id"].isin(active_users)].copy()

print("after active user filter:", df.shape)
print("active users:", df["user_id"].nunique())

print("Sampling users...")

sample_users = (
    pd.Series(df["user_id"].unique())
    .sample(
        n=min(N_USERS, df["user_id"].nunique()),
        random_state=42,
    )
    .values
)

df = df[df["user_id"].isin(sample_users)].copy()

print("final:", df.shape)
print("users:", df["user_id"].nunique())
print("items:", df["item_id"].nunique())

df.to_parquet(SAVE_PATH)

print("saved to", SAVE_PATH)