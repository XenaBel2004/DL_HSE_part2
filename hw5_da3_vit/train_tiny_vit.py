import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt

DATA_DIR = Path("data_depth")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_classes=2, embed_dim=64, depth=2, heads=4, mlp_dim=128):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)  # self-attention is inside TransformerEncoderLayer
        x = self.norm(x[:, 0])
        return self.head(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("data_depth/ not found. Run python run_da3_generate_depth.py first.")

    device = get_device()
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder(DATA_DIR / "train", transform=transform)
    val_data = ImageFolder(DATA_DIR / "val", transform=transform)
    print("Classes:", train_data.classes)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

    model = TinyViT(num_classes=len(train_data.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    losses = []
    val_accs = []

    for epoch in range(8):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        losses.append(avg_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), OUT_DIR / "tiny_vit_on_da3_depth.pth")

    plt.figure()
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(OUT_DIR / "loss.png", dpi=150)

    plt.figure()
    plt.plot(val_accs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy")
    plt.savefig(OUT_DIR / "accuracy.png", dpi=150)

    print("Saved outputs to outputs/")


if __name__ == "__main__":
    main()
