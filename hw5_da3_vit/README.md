# HW5: DA3 generation + TinyViT training

Pipeline:

1. Generate a small RGB dataset with two classes: circles and squares.
2. Use Depth Anything 3 from the official repository to generate depth maps for these RGB images.
3. Save depth maps in ImageFolder format.
4. Train a small custom Vision Transformer with self-attention on the DA3-generated depth maps.

## Run on MacBook

```bash
cd /Users/kseniia/PycharmProjects/DL2/hw5_da3_vit
python3 -m venv da3_vit_env
source da3_vit_env/bin/activate
pip install --upgrade pip
pip install -r requirements_vit.txt
```

Install the official Depth Anything 3 repo separately. If Python 3.13 causes problems, use Python 3.12.

```bash
cd /Users/kseniia/PycharmProjects/DL2/hw5_da3_vit/depth-anything-3/depth-anything-3
pip install -e . --no-deps
```

Then return to this project folder:

```bash
cd /Users/kseniia/PycharmProjects/DL2/hw5_da3_vit/hw5_da3_to_vit
python make_rgb_dataset.py
python run_da3_generate_depth.py
python train_tiny_vit.py
```

Expected outputs:

```text
data_rgb/
data_depth/
outputs/tiny_vit_on_da3_depth.pth
outputs/loss.png
outputs/accuracy.png
```
