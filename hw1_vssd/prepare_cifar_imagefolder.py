from torchvision.datasets import CIFAR10
import os

root = "cifar_imagefolder"

train_data = CIFAR10(root="data", train=True, download=True)
test_data = CIFAR10(root="data", train=False, download=True)

classes = train_data.classes

for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(f"{root}/{split}/{cls}", exist_ok=True)

for i, (img, label) in enumerate(train_data):
    if i < 3000:
        split = "train"
    elif i < 3600:
        split = "val"
    else:
        break

    cls = classes[label]
    img.save(f"{root}/{split}/{cls}/{i}.png")