"""Thông số dùng chung cho toàn bộ project (cả phần tấn công và phòng thủ).

Mọi notebook đều import từ đây, KHÔNG hard-code lại các giá trị này trong notebook.
"""
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/AdvML_Project")

# Trên Colab có Drive -> lưu checkpoint lên Drive để không mất khi runtime bị ngắt.
if IN_COLAB and DRIVE_PROJECT_DIR.is_dir():
    CHECKPOINT_DIR = DRIVE_PROJECT_DIR / "checkpoints"
else:
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

BASELINE_CKPT = "baseline_resnet18.pth"

# ---------------------------------------------------------------------------
# Huấn luyện
# ---------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Tấn công (epsilon tính trên thang pixel [0, 1])
# ---------------------------------------------------------------------------
EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]
PGD_STEPS = 10
PGD_ALPHA = 2 / 255
N_EVAL = 1000  # số ảnh test dùng khi đánh giá tấn công

# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = len(CLASSES)


def get_device() -> torch.device:
    """Trả về cuda nếu có GPU, ngược lại mps (Apple) hoặc cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
