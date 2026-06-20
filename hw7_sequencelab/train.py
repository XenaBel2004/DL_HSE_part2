import os
import time
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

seed = 42
random.seed(seed)
torch.manual_seed(seed)


transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset = Subset(train_dataset, list(range(12000)))
test_dataset = Subset(test_dataset, list(range(3000)))

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


class BaselineMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class WaveMixerBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

        self.mix = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim
        )

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        x = self.norm(x)

        v = self.value(x)
        g = torch.sigmoid(self.gate(x))

        # Conv1d ожидает формат B, C, T
        mixed = self.mix(v.transpose(1, 2)).transpose(1, 2)

        out = g * mixed + (1 - g) * v
        out = self.proj(out)

        return residual + out


class WaveSequenceModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.embed = nn.Linear(28, dim)

        self.block1 = WaveMixerBlock(dim)
        self.block2 = WaveMixerBlock(dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # x: B, 1, 28, 28
        x = x.squeeze(1)          # B, 28, 28
        x = self.embed(x)         # B, 28, dim

        x = self.block1(x)
        x = self.block2(x)

        x = x.mean(dim=1)

        return self.classifier(x)


def train_one_model(model, name, epochs=8, lr=1e-3):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"{name} | Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")

    train_time = time.time() - start_time

    torch.save(model.state_dict(), f"results/{name}.pt")

    return model, train_time


def evaluate(model):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1


baseline, baseline_time = train_one_model(
    BaselineMLP(),
    name="baseline_mlp",
    epochs=8,
    lr=1e-3
)

wave_model, wave_time = train_one_model(
    WaveSequenceModel(),
    name="wave_sequence_model",
    epochs=8,
    lr=1e-3
)

baseline_acc, baseline_f1 = evaluate(baseline)
wave_acc, wave_f1 = evaluate(wave_model)

df = pd.DataFrame({
    "model": ["Baseline MLP", "WAVE-like Sequence Model"],
    "accuracy": [baseline_acc, wave_acc],
    "f1_macro": [baseline_f1, wave_f1],
    "train_time_sec": [baseline_time, wave_time]
})

print(df)

df.to_csv("results/final_results.csv", index=False)

print("Saved results to results/final_results.csv")