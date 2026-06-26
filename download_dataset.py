from huggingface_hub import hf_hub_download
import os

os.makedirs("data", exist_ok=True)

path = hf_hub_download(
    repo_id="deepvk/VK-LSVD",
    repo_type="dataset",
    filename="interactions/train/week_00.parquet",
    local_dir="data",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded path:", path)
print("File exists:", os.path.exists(path))
print("File size GB:", os.path.getsize(path) / 1024**3)