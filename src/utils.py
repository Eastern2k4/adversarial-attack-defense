"""Hàm tiện ích: seed, train/evaluate, lưu/khôi phục checkpoint."""
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import SEED, get_device


def set_seed(seed=SEED, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True tái lập chính xác hơn nhưng chậm hơn trên GPU.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def train_one_epoch(model, loader, optimizer, criterion=None, device=None, desc="train"):
    """Huấn luyện 1 epoch. Trả về (loss trung bình, accuracy %)."""
    device = device or get_device()
    criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{total_loss / total:.3f}", acc=f"{100 * correct / total:.2f}")
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def predict(model, loader, device=None):
    """Trả về (preds, labels) dạng tensor CPU cho toàn bộ loader."""
    device = device or get_device()
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device, non_blocking=True)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


@torch.no_grad()
def evaluate(model, loader, device=None, criterion=None, return_loss=False):
    """Accuracy (%) trên loader. return_loss=True -> trả về (loss, acc)."""
    device = device or get_device()
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = 100.0 * correct / total
    return (total_loss / total, acc) if return_loss else acc


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, best_acc=0.0, **extra):
    """Lưu trạng thái huấn luyện để resume. Ghi ra file tạm rồi đổi tên,
    tránh hỏng checkpoint nếu Colab bị ngắt giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    """Nạp checkpoint vào model (và optimizer/scheduler nếu truyền vào).
    Trả về dict checkpoint (chứa epoch, best_acc và các khoá thêm như history).
    Chấp nhận cả file chỉ chứa state_dict của model."""
    map_location = map_location or get_device()
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in ckpt:  # file chỉ có state_dict
        ckpt = {"model": ckpt, "epoch": 0, "best_acc": 0.0}
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
