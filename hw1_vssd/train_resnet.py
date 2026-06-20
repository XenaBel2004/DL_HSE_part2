import os
import time

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-4
OUT_DIR = "results"

TRAIN_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 2000

os.makedirs(OUT_DIR, exist_ok=True)


def get_loaders():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    full_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )

    full_train, _ = random_split(
        full_train,
        [TRAIN_SUBSET_SIZE, len(full_train) - TRAIN_SUBSET_SIZE],
        generator=torch.Generator().manual_seed(42)
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

    test_dataset, _ = random_split(
        test_dataset,
        [TEST_SUBSET_SIZE, len(test_dataset) - TEST_SUBSET_SIZE],
        generator=torch.Generator().manual_seed(42)
    )

    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def train_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return avg_loss, acc, f1


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return avg_loss, acc, f1


def plot_history(history):
    df = pd.DataFrame(history)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ResNet18 loss")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/resnet18_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train")
    plt.plot(df["epoch"], df["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("ResNet18 accuracy")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/resnet18_accuracy.png", dpi=200)
    plt.close()


def main():
    print("Device:", DEVICE)

    train_loader, val_loader, test_loader = get_loaders()

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        }

        history.append(row)
        print(row)

    train_time = time.time() - start_time

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)

    pd.DataFrame(history).to_csv(f"{OUT_DIR}/resnet18_history.csv", index=False)

    result_df = pd.DataFrame([{
        "model": "ResNet18",
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time_sec": train_time
    }])

    result_df.to_csv(f"{OUT_DIR}/resnet18_result.csv", index=False)

    plot_history(history)

    torch.save(model.state_dict(), f"{OUT_DIR}/resnet18.pt")

    print(result_df)


if __name__ == "__main__":
    main()