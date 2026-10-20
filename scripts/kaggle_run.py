"""Kaggle GPU job: resume fine-tuning the Monitor Nepal U-Net from the uploaded checkpoint (epoch 2 -> 12)."""
import glob
import os
import shutil
import subprocess
import sys

src = os.path.dirname(glob.glob("/kaggle/input/**/finetune_nepal.py", recursive=True)[0])
work = "/kaggle/working"
os.makedirs(f"{work}/out", exist_ok=True)
for name in ("finetune_nepal.py", "unet_torch.py"):
    shutil.copy(f"{src}/{name}", f"{work}/{name}")
checkpoint = f"{work}/out/finetune_checkpoint.pt"
if not os.path.exists(checkpoint):
    shutil.copy(f"{src}/finetune_checkpoint.pt", checkpoint)
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
subprocess.run([sys.executable, "-u", f"{work}/finetune_nepal.py", "--data", src, "--out", f"{work}/out",
                "--epochs", "12", "--batch", "16", "--no-export", "--keras", "unused.keras"], check=True)
