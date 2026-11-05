"""Preparation of reference data: CIFAR-10 test images, FID stats, fixed noise set.

Creates:
  data/fixed_noise.pt            — 1000 фиксированных начальных шумов (seed 2026)
  data/cifar10_test/*.png        — 10000 изображений CIFAR-10 test (референс для FID)
  data/fid_stats_cifar10_test.npz — mu/sigma референса в пространстве InceptionV3
"""
import io
import os

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)

# ---------------------------------------------------------------- fixed noise
# Device-independent: CPU generator, fixed seed -> identical noise everywhere.
N_NOISE = 1000
g = torch.Generator(device="cpu").manual_seed(2026)
noise = torch.randn(N_NOISE, 3, 32, 32, generator=g)
torch.save({"noise": noise, "seed": 2026, "n": N_NOISE}, os.path.join(DATA, "fixed_noise.pt"))
print("fixed noise saved:", tuple(noise.shape))

# ------------------------------------------------------------- CIFAR-10 test
test_parquet = os.path.join(DATA, "raw", "cifar10_test.parquet")
out_dir = os.path.join(DATA, "cifar10_test")
if not os.path.exists(test_parquet):
    import urllib.request

    # uoft-cs/cifar10 — публичное зеркало CIFAR-10 на HuggingFace (parquet)
    url = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"
    urllib.request.urlretrieve(url, test_parquet)
os.makedirs(out_dir, exist_ok=True)
import pyarrow.parquet as pq

table = pq.read_table(test_parquet)
for i in range(table.num_rows):
    img = Image.open(io.BytesIO(table["img"][i]["bytes"].as_py())).convert("RGB")
    img.save(os.path.join(out_dir, f"{i:05d}.png"))
print("CIFAR-10 test saved:", table.num_rows, "images")

# ------------------------------------------------------------- FID reference
from pytorch_fid.fid_score import compute_statistics_of_path
from pytorch_fid.inception import InceptionV3

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
mu, sigma = compute_statistics_of_path(out_dir, model, batch_size=200, dims=2048, device=device, num_workers=0)
np.savez(os.path.join(DATA, "fid_stats_cifar10_test.npz"), mu=mu, sigma=sigma)
print("FID reference stats saved; device:", device)
