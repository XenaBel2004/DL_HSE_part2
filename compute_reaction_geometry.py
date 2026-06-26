import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import permutations

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


REACTIONS = [
    "like",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
]


class ReactionDataset(Dataset):
    def __init__(self, users, item_embs, y):
        self.users = torch.tensor(users, dtype=torch.long)
        self.item_embs = torch.tensor(item_embs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.item_embs[idx], self.y[idx]


class MultiReactionMLP(nn.Module):
    def __init__(self, n_users, item_dim=64, user_dim=32, n_reactions=5):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, user_dim)

        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_reactions),
        )

    def forward(self, users, item_embs):
        u = self.user_emb(users)
        x = torch.cat([u, item_embs], dim=1)
        return self.net(x)


def train_reaction_model(df, device):
    X = np.stack(df["embedding"].values).astype("float32")
    users = df["user_idx"].values.astype("int64")
    y = df[REACTIONS].astype(float).values.astype("float32")

    dataset = ReactionDataset(users, X, y)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = MultiReactionMLP(
        n_users=df["user_idx"].nunique(),
        item_dim=64,
        user_dim=32,
        n_reactions=len(REACTIONS),
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for users_batch, item_batch, y_batch in loader:
            users_batch = users_batch.to(device)
            item_batch = item_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(users_batch, item_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss={total_loss / len(loader):.4f}")

    return model


def compute_gradient_matrix(model, user_idx, item_sample, device):
    model.eval()

    x = torch.tensor(
        item_sample,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    users = torch.full(
        (len(item_sample),),
        int(user_idx),
        dtype=torch.long,
        device=device,
    )

    logits = model(users, x)
    probs = torch.sigmoid(logits)

    m = len(REACTIONS)
    D = np.zeros((m, m), dtype=np.float32)

    for i in range(m):
        for j in range(i + 1, m):
            h = probs[:, i] - probs[:, j]

            grad = torch.autograd.grad(
                outputs=h.sum(),
                inputs=x,
                retain_graph=True,
                create_graph=False,
            )[0]

            grad_norm = torch.norm(grad, dim=1)
            lip = grad_norm.max().item()

            D[i, j] = lip
            D[j, i] = lip

    return D

def gromov_distance(D1, D2):
    n = D1.shape[0]

    best = np.inf

    for perm in permutations(range(n)):
        perm = np.array(perm)

        D2_perm = D2[np.ix_(perm, perm)]

        distortion = np.max(
            np.abs(D1 - D2_perm)
        )

        best = min(best, distortion)

    return 0.5 * best


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_parquet("data/vk_working_subset.parquet")

    train_idx = np.load("data/train_idx.npy")
    df = df.iloc[train_idx].copy()

    user_encoder = LabelEncoder()
    df["user_idx"] = user_encoder.fit_transform(df["user_id"])

    print("Dataset:", df.shape)
    print("Users:", df["user_idx"].nunique())
    print("Items:", df["item_id"].nunique())

    for col in REACTIONS:
        print(col, "rate:", df[col].mean())

    model = train_reaction_model(df, device)

    X_items = np.stack(df["embedding"].values).astype("float32")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(
        len(X_items),
        size=min(1000, len(X_items)),
        replace=False,
    )
    item_sample = X_items[sample_idx]

    counts = df["user_idx"].value_counts()
    good_users = counts[counts >= 5].index.to_numpy()

    print("Good users:", len(good_users))

    user_matrices = {}

    for u in tqdm(good_users):
        D_u = compute_gradient_matrix(
            model=model,
            user_idx=int(u),
            item_sample=item_sample,
            device=device,
        )
        user_matrices[int(u)] = D_u

    np.save(
        "data/user_reaction_matrices.npy",
        user_matrices,
        allow_pickle=True,
    )

    print("Saved to data/user_reaction_matrices.npy")

    users = list(user_matrices.keys())

    distances = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u = users[i]
            v = users[j]

            frob = np.linalg.norm(
                user_matrices[u] - user_matrices[v]
            )

            gh = gromov_distance(
                user_matrices[u],
                user_matrices[v]
            )

            distances.append({
                "user_1": u,
                "user_2": v,
                "frobenius_distance": frob,
                "gromov_distance": gh,
            })

    dist_df = pd.DataFrame(distances)
    dist_df.to_csv("data/user_distances.csv", index=False)

    print("\nGromov distances:")
    print(dist_df["gromov_distance"].describe())

    print("\nExample matrix:")
    print(user_matrices[users[0]])


if __name__ == "__main__":
    main()