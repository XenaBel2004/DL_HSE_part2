import os
import math
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True
)


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


model = CNNAttention().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_time = time.time()

epochs = 20

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
    print(f"Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

train_time = time.time() - start_time

torch.save(model.state_dict(), "results/mhla.pt")

with open("results/mhla_time.txt", "w") as f:
    f.write(str(train_time))

print("Saved model to results/mhla.pt")
print(f"Training time: {train_time:.2f} sec")