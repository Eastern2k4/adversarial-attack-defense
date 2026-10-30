"""Tấn công đối kháng L∞: FGSM và PGD.

Quy ước (xem docs/INTERFACE.md):
- x: tensor [N,3,32,32] trong [0,1], cùng device với model; y: nhãn đúng
  (hoặc nhãn đích nếu targeted=True — khi đó đi theo chiều GIẢM loss).
- Trả về x_adv đã detach, clamp [0,1], ||x_adv - x||∞ ≤ eps.
- Không đổi model.train()/eval(), không tích gradient vào tham số mô hình
  (gradient lấy theo x bằng torch.autograd.grad).
- Người gọi tự đặt model.eval(); Normalize đã nằm bên trong mô hình.
"""
import math
import operator

import torch
import torch.nn.functional as F


def _validate_budget(name, value):
    """Ngân sách nhiễu và độ dài bước phải hữu hạn, không âm."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _input_grad(model, x, y):
    """Tính gradient cross-entropy theo một bản sao độc lập của ảnh.

    enable_grad cho phép gọi tấn công trong torch.no_grad(). autograd.grad
    chỉ trả gradient theo ảnh, giữ nguyên cả gradient trọng số đã có từ trước.
    """
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        (grad,) = torch.autograd.grad(loss, x, create_graph=False, retain_graph=False)
    return grad.detach()


def fgsm(model, x, y, eps, targeted=False):
    """FGSM: một bước theo dấu gradient, giới hạn nhiễu L∞ bằng eps.

    Untargeted: tăng cross-entropy với nhãn thật y.
    Targeted: giảm cross-entropy với nhãn đích y do người gọi cung cấp.
    Trả ảnh đã detach, cùng shape/device/dtype với x và nằm trong [0,1].
    Không sửa x, chế độ mô hình hoặc .grad của ảnh/trọng số.
    """
    _validate_budget("eps", eps)
    x = x.detach()
    if eps == 0 or x.numel() == 0:
        return x.clone()

    grad = _input_grad(model, x, y)
    step = -eps if targeted else eps
    x_adv = x + step * grad.sign()
    # Một bước cỡ eps đã thỏa L∞; clamp chỉ giảm độ lệch khi x ở [0,1].
    return x_adv.clamp(0.0, 1.0).detach()


def pgd(model, x, y, eps, alpha, steps, random_start=True, targeted=False):
    """Projected Gradient Descent L∞ (Madry et al., 2018).

    Mỗi bước tăng loss với nhãn thật; targeted=True thì giảm loss với nhãn đích.
    Chiếu sau mỗi bước vào giao của [x-eps, x+eps] và [0,1].
    random_start=True: lấy nhiễu đều trong [-eps, eps], rồi clamp ảnh về [0,1].
    Dùng RNG của PyTorch; người gọi đặt seed để tái lập khởi tạo.

    steps phải là số nguyên không âm. steps=0 trả ảnh khởi tạo (có thể có
    nhiễu nếu random_start=True). eps=0 trả bản sao ảnh gốc, không gọi model.
    Trả ảnh đã detach, không sửa x, chế độ mô hình hoặc .grad của ảnh/trọng số.
    """
    _validate_budget("eps", eps)
    _validate_budget("alpha", alpha)
    steps = operator.index(steps)
    if steps < 0:
        raise ValueError("steps must be a non-negative integer.")

    x = x.detach()
    if eps == 0 or x.numel() == 0:
        return x.clone()

    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x.clone()

    if steps == 0 or alpha == 0:
        return x_adv.detach()

    # Hai biên luôn tính từ ảnh gốc, không từ ảnh của vòng lặp trước.
    lower = (x - eps).clamp(0.0, 1.0)
    upper = (x + eps).clamp(0.0, 1.0)
    step = -alpha if targeted else alpha
    for _ in range(steps):
        grad = _input_grad(model, x_adv, y)
        x_adv = x_adv + step * grad.sign()
        x_adv = torch.maximum(torch.minimum(x_adv, upper), lower).detach()
    return x_adv.detach()
