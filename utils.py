import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(
    interactions_path: str,
    embeddings_path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
):
    interactions = pd.read_parquet(interactions_path)
    item_embs = pd.read_parquet(embeddings_path)

    interactions = interactions.merge(item_embs, on=item_col, how="inner")

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    interactions["user_idx"] = user_encoder.fit_transform(interactions[user_col])
    interactions["item_idx"] = item_encoder.fit_transform(interactions[item_col])

    return interactions, user_encoder, item_encoder


def sample_dataset(
    df: pd.DataFrame,
    n_users: int = 1000,
    n_items: int = 5000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    users = df["user_idx"].unique()
    users = rng.choice(users, size=min(n_users, len(users)), replace=False)

    df = df[df["user_idx"].isin(users)].copy()

    top_items = df["item_idx"].value_counts().head(n_items).index
    df = df[df["item_idx"].isin(top_items)].copy()

    return df.reset_index(drop=True)


def get_embedding_matrix(df: pd.DataFrame, emb_col: str = "embedding"):
    X = np.stack(df[emb_col].values).astype("float32")
    return X


def make_binary_target(df, col):
    return df[col].fillna(0).astype("float32").values


def make_watch_target(df, col="timespent"):
    y = df[col].fillna(0).astype("float32").values
    return np.log1p(y)