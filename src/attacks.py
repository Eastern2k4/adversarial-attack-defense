# Bản tạm do thành viên phòng thủ viết để không bị chặn tiến độ. Thành viên tấn công có thể sửa/viết lại nhưng GIỮ NGUYÊN chữ ký hàm (xem docs/INTERFACE.md).
"""Tấn công đối kháng L∞: FGSM và PGD.

Quy ước (xem docs/INTERFACE.md):
- x: tensor [N,3,32,32] trong [0,1], cùng device với model; y: nhãn đúng
  (hoặc nhãn đích nếu targeted=True — khi đó đi theo chiều GIẢM loss).
- Trả về x_adv đã detach, clamp [0,1], ||x_adv - x||∞ ≤ eps.
- Không đổi model.train()/eval(), không tích gradient vào tham số mô hình
  (gradient lấy theo x bằng torch.autograd.grad).
"""
import torch
import torch.nn.functional as F


def _input_grad(model, x, y):
    """Gradient của cross-entropy theo đầu vào x (không chạm vào .grad của tham số).
    enable_grad để vẫn chạy được khi bị gọi bên trong torch.no_grad()."""
    with torch.enable_grad():
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(model, x, y, eps, targeted=False):
    """Fast Gradient Sign Method (Goodfellow et al., 2015): x + eps * sign(∇x L)."""
    x = x.detach()
    grad = _input_grad(model, x, y)
    step = -eps if targeted else eps
    x_adv = x + step * grad.sign()
    return x_adv.clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, alpha, steps, random_start=True, targeted=False):
    """Projected Gradient Descent L∞ (Madry et al., 2018).

    Mỗi bước: x_adv += alpha * sign(∇x L), chiếu về hình cầu L∞ bán kính eps quanh x
    và clamp về [0,1]. random_start: khởi tạo nhiễu đều trong [-eps, eps].
    """
    x = x.detach()
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    step = -alpha if targeted else alpha
    for _ in range(steps):
        grad = _input_grad(model, x_adv, y)
        x_adv = x_adv.detach() + step * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
    return x_adv.detach()
