import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import (
    load_data,
    sample_dataset,
    get_embedding_matrix,
    make_binary_target,
    make_watch_target,
)
from models import ReactionModel
from geometry import (
    compute_user_reaction_matrix,
    frobenius_user_distance,
    gromov_wasserstein_distance,
)


def train_reaction_model(
    df,
    reaction_col,
    target_type,
    emb_col="embedding",
    epochs=3,
    batch_size=512,
    lr=1e-3,
    device="cpu",
):
    X = get_embedding_matrix(df, emb_col=emb_col)
    users = df["user_idx"].values.astype("int64")

    if target_type == "binary":
        y = make_binary_target(df, reaction_col)
        loss_fn = nn.BCEWithLogitsLoss()
    elif target_type == "watch":
        y = make_watch_target(df, reaction_col)
        loss_fn = nn.MSELoss()
    else:
        raise ValueError("target_type должен быть binary или watch")

    X = torch.tensor(X, dtype=torch.float32)
    users = torch.tensor(users, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(users, X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_users = df["user_idx"].nunique()
    emb_dim = X.shape[1]

    model = ReactionModel(
        emb_dim=emb_dim,
        n_users=n_users,
        hidden_dim=128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for user_batch, x_batch, y_batch in loader:
            user_batch = user_batch.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(user_batch, x_batch)

            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"{reaction_col}, epoch {epoch + 1}, "
            f"loss={total_loss / len(loader):.4f}"
        )

    return model


def main():
    device = "cpu"

    interactions_path = "data/interactions.parquet"
    embeddings_path = "data/item_embeddings.parquet"

    df, user_encoder, item_encoder = load_data(
        interactions_path=interactions_path,
        embeddings_path=embeddings_path,
        user_col="user_id",
        item_col="item_id",
    )

    df = sample_dataset(
        df,
        n_users=1000,
        n_items=5000,
    )

    print("Dataset shape:", df.shape)
    print("Users:", df["user_idx"].nunique())
    print("Items:", df["item_idx"].nunique())

    reactions = {
        "like": "binary",
        "share": "binary",
        "bookmark": "binary",
        "open_comments": "binary",
        "timespent": "watch",
    }

    models = {}

    for reaction, target_type in reactions.items():
        if reaction not in df.columns:
            print(f"Skip {reaction}: нет такой колонки")
            continue

        print(f"\nTraining model for {reaction}")

        model = train_reaction_model(
            df=df,
            reaction_col=reaction,
            target_type=target_type,
            emb_col="embedding",
            epochs=3,
            batch_size=512,
            lr=1e-3,
            device=device,
        )

        models[reaction] = model

    item_embs = get_embedding_matrix(df, emb_col="embedding")

    # Берём небольшую выборку item embeddings для оценки Lipschitz
    rng = np.random.default_rng(42)
    idx = rng.choice(
        len(item_embs),
        size=min(1000, len(item_embs)),
        replace=False,
    )
    item_embs_sample = item_embs[idx]

    users = df["user_idx"].unique()[:20]

    user_spaces = {}

    for u in tqdm(users):
        D_u, reaction_names = compute_user_reaction_matrix(
            user_id=int(u),
            models=models,
            item_embs=item_embs_sample,
            device=device,
        )

        user_spaces[int(u)] = D_u

    u1 = int(users[0])
    u2 = int(users[1])

    D1 = user_spaces[u1]
    D2 = user_spaces[u2]

    print("\nReaction names:")
    print(reaction_names)

    print(f"\nMatrix for user {u1}:")
    print(D1)

    print(f"\nMatrix for user {u2}:")
    print(D2)

    print("\nFrobenius distance:")
    print(frobenius_user_distance(D1, D2))

    print("\nGromov-Wasserstein distance:")
    print(gromov_wasserstein_distance(D1, D2))


if __name__ == "__main__":
    main()