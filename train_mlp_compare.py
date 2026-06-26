# train_mlp_compare.py

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
import torch.nn as nn
import random

from sklearn.metrics import average_precision_score

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

def best_f1_score(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1)

    return float(f1[best_idx])

class VKDataset(Dataset):
    def __init__(self, users, items_emb, y, geometry=None):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items_emb = torch.tensor(items_emb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if geometry is not None:
            self.geometry = torch.tensor(geometry, dtype=torch.float32)
        else:
            self.geometry = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.geometry is None:
            return self.users[idx], self.items_emb[idx], self.y[idx]
        return self.users[idx], self.items_emb[idx], self.geometry[idx], self.y[idx]

class BaselineMLP(nn.Module):
    def __init__(self, n_users, item_dim=64, user_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, user_dim)

        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item_emb):
        u = self.user_emb(user)
        x = torch.cat([u, item_emb], dim=1)
        return self.net(x).squeeze(1)


class GeometryMLP(nn.Module):
    def __init__(self, n_users, item_dim=64, geom_dim=25, user_dim=32):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, user_dim)

        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(user_dim + item_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item_emb, geometry):
        u = self.user_emb(user)
        g = self.geom_encoder(geometry)
        x = torch.cat([u, item_emb, g], dim=1)
        return self.net(x).squeeze(1)

def train_epoch_baseline(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for users, item_emb, y in loader:
        users = users.to(device)
        item_emb = item_emb.to(device)
        y = y.to(device)

        logits = model(users, item_emb)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch_geometry(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for users, item_emb, geometry, y in loader:
        users = users.to(device)
        item_emb = item_emb.to(device)
        geometry = geometry.to(device)
        y = y.to(device)

        logits = model(users, item_emb, geometry)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_baseline(model, loader, device):
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for users, item_emb, y in loader:
            users = users.to(device)
            item_emb = item_emb.to(device)

            logits = model(users, item_emb)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_prob.extend(probs)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": best_f1_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def evaluate_geometry(model, loader, device):
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for users, item_emb, geometry, y in loader:
            users = users.to(device)
            item_emb = item_emb.to(device)
            geometry = geometry.to(device)

            logits = model(users, item_emb, geometry)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_prob.extend(probs)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": best_f1_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    device = "cpu"
    print("Device:", device)

    df = pd.read_parquet("data/vk_working_subset.parquet")

    print("Dataset:", df.shape)
    print("Like rate:", df["like"].mean())

    user_encoder = LabelEncoder()
    df["user_idx"] = user_encoder.fit_transform(df["user_id"])

    n_users = df["user_idx"].nunique()

    X_item = np.stack(df["embedding"].values).astype("float32")
    users = df["user_idx"].values.astype("int64")
    y = df["like"].astype(int).values.astype("float32")

    user_matrices = np.load(
        "data/user_reaction_matrices.npy",
        allow_pickle=True
    ).item()

    geometry = []

    for u in users:
        if int(u) in user_matrices:
            geometry.append(user_matrices[int(u)].flatten())
        else:
            geometry.append(np.zeros(25, dtype="float32"))

    geometry = np.stack(geometry).astype("float32")
    geom_norms = np.linalg.norm(geometry, axis=1)
    print("Rows with non-zero geometry:", (geom_norms > 0).sum())
    print("Total rows:", len(geometry))
    print("Geometry coverage:", (geom_norms > 0).mean())

    train_idx = np.load("data/train_idx.npy")
    test_idx = np.load("data/test_idx.npy")

    train_base = VKDataset(
        users[train_idx],
        X_item[train_idx],
        y[train_idx],
    )

    test_base = VKDataset(
        users[test_idx],
        X_item[test_idx],
        y[test_idx],
    )

    train_geom = VKDataset(
        users[train_idx],
        X_item[train_idx],
        y[train_idx],
        geometry=geometry[train_idx],
    )

    test_geom = VKDataset(
        users[test_idx],
        X_item[test_idx],
        y[test_idx],
        geometry=geometry[test_idx],
    )

    train_base_loader = DataLoader(train_base, batch_size=512, shuffle=True)
    test_base_loader = DataLoader(test_base, batch_size=512)

    train_geom_loader = DataLoader(train_geom, batch_size=512, shuffle=True)
    test_geom_loader = DataLoader(test_geom, batch_size=512)

    loss_fn = nn.BCEWithLogitsLoss()

    print("\nTraining Baseline MLP")

    baseline = BaselineMLP(
        n_users=n_users,
        item_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        baseline.parameters(),
        lr=1e-3,
    )

    for epoch in range(15):
        loss = train_epoch_baseline(
            baseline,
            train_base_loader,
            optimizer,
            loss_fn,
            device,
        )

        metrics = evaluate_baseline(
            baseline,
            test_base_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={loss:.4f}, "
            f"auc={metrics['auc']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"acc={metrics['accuracy']:.4f}"
        )

    baseline_metrics = evaluate_baseline(
        baseline,
        test_base_loader,
        device,
    )


    print("\nTraining Geometry MLP")

    geometry_model = GeometryMLP(
        n_users=n_users,
        item_dim=64,
        geom_dim=25,
    ).to(device)

    optimizer = torch.optim.Adam(
        geometry_model.parameters(),
        lr=1e-3,
    )

    for epoch in range(15):
        loss = train_epoch_geometry(
            geometry_model,
            train_geom_loader,
            optimizer,
            loss_fn,
            device,
        )

        metrics = evaluate_geometry(
            geometry_model,
            test_geom_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={loss:.4f}, "
            f"auc={metrics['auc']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"acc={metrics['accuracy']:.4f}"
        )

    geometry_metrics = evaluate_geometry(
        geometry_model,
        test_geom_loader,
        device,
    )


    results = pd.DataFrame([
        {
            "model": "Baseline MLP",
            **baseline_metrics,
        },
        {
            "model": "MLP + Reaction Geometry",
            **geometry_metrics,
        },
    ])

    print("\nFinal results:")
    print(results)

    torch.save(
        baseline.state_dict(),
        "data/baseline_mlp.pt"
    )

    torch.save(
        geometry_model.state_dict(),
        "data/geometry_mlp.pt"
    )

    np.save(
        "data/user_encoder_classes.npy",
        user_encoder.classes_,
        allow_pickle=True
    )

    print("Saved models and user encoder")

    results.to_csv("data/mlp_results.csv", index=False)
    print("\nSaved to data/mlp_results.csv")


if __name__ == "__main__":
    main()