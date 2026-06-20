import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

device = "mps" if torch.backends.mps.is_available() else "cpu"

transform = transforms.ToTensor()

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x

        x = self.norm(x)

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attn = torch.softmax(
            Q @ K.transpose(-2, -1) / math.sqrt(x.shape[-1]),
            dim=-1
        )

        out = attn @ V
        out = self.proj(out)

        return residual + out


class CNNAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.attn = AttentionBlock(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv(x)

        B, C, H, W = x.shape

        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        x = self.attn(x)

        x = x.mean(dim=1)

        return self.classifier(x)


def evaluate(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return acc, f1


baseline_acc, baseline_f1 = evaluate(
    SimpleCNN(),
    "results/baseline.pt"
)

mhla_acc, mhla_f1 = evaluate(
    CNNAttention(),
    "results/mhla.pt"
)

df = pd.DataFrame({
    "model": ["Baseline CNN", "CNN + Attention"],
    "accuracy": [baseline_acc, mhla_acc],
    "f1_macro": [baseline_f1, mhla_f1]
})

print(df)

df.to_csv(
    "results/final_results.csv",
    index=False
)