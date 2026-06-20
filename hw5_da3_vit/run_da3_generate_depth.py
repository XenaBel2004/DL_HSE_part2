"""
This script uses Depth Anything 3 from the official repository to generate depth maps.
It reads RGB images from data_rgb/ and saves DA3-generated depth images to data_depth/.

Run it after installing the official DA3 repository in editable mode.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch

try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print("Cannot import depth_anything_3.")
    print("Install the official repo first, for example:")
    print("  cd /path/to/depth-anything-3")
    print("  pip install -e . --no-deps")
    raise e

RGB_DIR = Path("data_rgb")
DEPTH_DIR = Path("data_depth")
MODEL_NAME = "depth-anything/DA3-SMALL"  # small version is better for MacBook


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.uint8)
    depth_norm = (depth - d_min) / (d_max - d_min)
    return (depth_norm * 255).clip(0, 255).astype(np.uint8)


def main():
    if not RGB_DIR.exists():
        print("data_rgb/ not found. Run python make_rgb_dataset.py first.")
        sys.exit(1)

    device = get_device()
    print("Device:", device)
    print("Loading DA3 model:", MODEL_NAME)

    model = DepthAnything3.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    for split_dir in sorted(RGB_DIR.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            cls = class_dir.name
            out_dir = DEPTH_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(class_dir.glob("*.png"))
            for img_path in image_paths:
                print("DA3 inference:", img_path)
                pred = model.inference([str(img_path)])
                depth = pred.depth[0]  # [H, W]

                depth_u8 = normalize_depth_to_uint8(depth)
                out_path = out_dir / img_path.name
                Image.fromarray(depth_u8).save(out_path)

                # also save raw depth for reproducibility
                np.save(out_path.with_suffix(".npy"), depth.astype(np.float32))

    print("Depth dataset saved to data_depth/")


if __name__ == "__main__":
    main()
