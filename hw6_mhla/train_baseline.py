import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "mps" if torch.backends.mps.is_available() else "cpu"

transform = transforms.ToTensor()

train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(64*8*8,10)

    def forward(self,x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x)

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()

    for x,y in train_loader:
        x,y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

torch.save(model.state_dict(),"results/baseline.pt")