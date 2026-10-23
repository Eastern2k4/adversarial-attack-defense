"""Phòng thủ: Adversarial Training (Fast-AT, PGD-AT) và tiền xử lý đầu vào (JPEG, giảm bit, Gaussian blur).

Mọi ảnh vào/ra đều là tensor [N,3,32,32] trong [0, 1] (Normalize nằm trong mô hình).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.io import decode_jpeg, encode_jpeg
from tqdm.auto import tqdm

from .attacks import pgd


# ---------------------------------------------------------------------------
# Adversarial training
# ---------------------------------------------------------------------------
def train_epoch_fast_at(model, loader, optimizer, scheduler, eps, alpha, device, desc="fast-AT"):
    """1 epoch Fast-AT ("Fast is better than free", Wong et al. 2020).

    Mỗi batch: delta ~ U[-eps, eps] -> 1 bước FGSM cỡ alpha -> chiếu về [-eps, eps] và [0,1].
    Nhiễu được sinh ở model.eval(), sau đó chuyển model.train() để cập nhật trọng số
    trên ảnh đối kháng. scheduler.step() gọi sau MỖI batch (OneCycleLR / cyclic LR).
    Trả về (loss trung bình, accuracy % trên ảnh đối kháng).
    """
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Sinh nhiễu: FGSM với khởi tạo ngẫu nhiên
        model.eval()
        delta = torch.empty_like(x).uniform_(-eps, eps)
        delta = ((x + delta).clamp(0.0, 1.0) - x).requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        (grad,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * grad.sign()).clamp(-eps, eps)
        x_adv = (x + delta).clamp(0.0, 1.0).detach()

        # Cập nhật trọng số
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{total_loss / total:.3f}", acc=f"{100 * correct / total:.2f}")
    return total_loss / total, 100.0 * correct / total


def train_epoch_pgd_at(model, loader, optimizer, eps, alpha, steps, device, desc="PGD-AT"):
    """1 epoch PGD-AT (Madry et al. 2018): huấn luyện trên ảnh PGD (random start) sinh bằng src.attacks.pgd.

    Nhiễu sinh ở model.eval(), cập nhật ở model.train(). Scheduler (nếu có) step theo epoch ở ngoài.
    Trả về (loss trung bình, accuracy % trên ảnh đối kháng).
    """
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        model.eval()
        x_adv = pgd(model, x, y, eps, alpha, steps, random_start=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=f"{total_loss / total:.3f}", acc=f"{100 * correct / total:.2f}")
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Tiền xử lý đầu vào
# ---------------------------------------------------------------------------
@torch.no_grad()
def jpeg_compress(x, quality):
    """Nén rồi giải nén JPEG từng ảnh (xử lý trên CPU), trả về cùng device với x. Không khả vi."""
    imgs = (x.detach().clamp(0.0, 1.0) * 255).round().to(torch.uint8).cpu()
    out = [decode_jpeg(encode_jpeg(img, quality=int(quality))) for img in imgs]
    return (torch.stack(out).float() / 255.0).to(x.device)


@torch.no_grad()
def reduce_bit_depth(x, bits):
    """Feature squeezing (Xu et al. 2018): lượng tử hoá mỗi kênh còn 2^bits mức. Không khả vi."""
    levels = 2 ** int(bits) - 1
    return torch.round(x.detach().clamp(0.0, 1.0) * levels) / levels


def gaussian_blur(x, sigma):
    """Làm mờ Gaussian, kernel = 2*ceil(3*sigma)+1. Khả vi, chạy trên device của x."""
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(x, kernel_size=[k, k], sigma=[float(sigma), float(sigma)]).clamp(0.0, 1.0)


class PreprocessedModel(nn.Module):
    """Bọc model với bước tiền xử lý g: forward(x) = model(g(x)).

    - bpda=False: forward = model(g(x)). Gradient chỉ đi qua g nếu g khả vi
      (blur thì được; JPEG/giảm bit thì không -> không dùng để tấn công gradient).
    - bpda=True : forward = model(x + (g(x) - x).detach()). Giá trị forward vẫn là model(g(x)),
      nhưng backward coi g là hàm đồng nhất (BPDA, Athalye et al. 2018) -> kẻ tấn công
      thích nghi tính được gradient xuyên qua bước tiền xử lý.
    """

    def __init__(self, model, preprocess, bpda=False):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.bpda = bpda

    def forward(self, x):
        if self.bpda:
            return self.model(x + (self.preprocess(x.detach()) - x).detach())
        return self.model(self.preprocess(x))
