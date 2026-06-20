from PIL import Image, ImageDraw
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path("data_rgb")
CLASSES = ["circles", "squares"]
IMG_SIZE = 224


def make_image(cls: str, save_path: Path):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (235, 235, 235))
    draw = ImageDraw.Draw(img)

    # background distractors
    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, IMG_SIZE - 40)
        y1 = random.randint(0, IMG_SIZE - 40)
        x2 = x1 + random.randint(15, 45)
        y2 = y1 + random.randint(15, 45)
        color = tuple(random.randint(150, 255) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], fill=color)

    # target object
    color = tuple(random.randint(30, 230) for _ in range(3))
    x1 = random.randint(45, 100)
    y1 = random.randint(45, 100)
    size = random.randint(55, 90)
    x2 = x1 + size
    y2 = y1 + size

    if cls == "circles":
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif cls == "squares":
        draw.rectangle([x1, y1, x2, y2], fill=color)
    else:
        raise ValueError(f"Unknown class: {cls}")

    img.save(save_path)


def main():
    for split, n_per_class in [("train", 60), ("val", 15)]:
        for cls in CLASSES:
            out_dir = BASE_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                make_image(cls, out_dir / f"{cls}_{i:03d}.png")

    print("RGB dataset saved to data_rgb/")
    print("train: 120 images, val: 30 images")


if __name__ == "__main__":
    main()
