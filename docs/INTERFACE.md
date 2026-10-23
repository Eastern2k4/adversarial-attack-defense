# Giao diện chung giữa phần Tấn công và phần Phòng thủ

Đây là giao diện **đã chốt** giữa 2 thành viên. **Không được đổi tên hàm/tham số.** Có thể sửa phần cài đặt bên trong, miễn giữ đúng quy ước dưới đây.

Quy ước chung của project: ảnh luôn nằm trong **[0, 1]**, `Normalize` nằm **bên trong** mô hình (`build_model()`), mọi thông số lấy từ `src/config.py`.

## 1. `src/attacks.py`

```python
fgsm(model, x, y, eps, targeted=False) -> x_adv
pgd(model, x, y, eps, alpha, steps, random_start=True, targeted=False) -> x_adv
```

Quy ước:

- `x` là tensor `[N,3,32,32]` trong `[0,1]`, **cùng device với model**.
- `y` là nhãn đúng (hoặc **nhãn đích** nếu `targeted=True`; targeted thì đi theo chiều **GIẢM** loss).
- Trả về `x_adv` **đã detach**, **clamp [0,1]**, `||x_adv - x||∞ ≤ eps` (`pgd` **chiếu về hình cầu eps sau mỗi bước**).
- Hàm **KHÔNG** đổi `model.train()`/`model.eval()` và **KHÔNG** tích gradient vào tham số mô hình (dùng `torch.autograd.grad` theo `x`).
- `random_start`: khởi tạo nhiễu **đều trong [-eps, eps]**.

> `src/attacks.py` hiện tại là **bản tạm** do thành viên phòng thủ viết để không bị chặn tiến độ. Thành viên tấn công có thể sửa/viết lại nhưng **giữ nguyên chữ ký hàm**.

Ví dụ:

```python
from src import config
from src.attacks import fgsm, pgd

model.eval()
x, y = x.to(device), y.to(device)
x_fgsm = fgsm(model, x, y, eps=8/255)
x_pgd  = pgd(model, x, y, eps=8/255, alpha=config.PGD_ALPHA, steps=config.PGD_STEPS)

# Targeted: đẩy mọi ảnh về lớp 3 ("cat")
y_target = torch.full_like(y, 3)
x_tgt = pgd(model, x, y_target, eps=8/255, alpha=2/255, steps=10, targeted=True)
```

## 2. `src/utils.py`: `accuracy_under_attack`

```python
accuracy_under_attack(model, x, y, attack_fn=None, batch_size=256) -> float  # accuracy %
```

- `attack_fn(model, x, y) -> x_adv`; `attack_fn=None` = đánh giá trên ảnh sạch.
- Hàm **tự chia batch**, **tự đưa dữ liệu lên device của model**, **đặt `model.eval()`**.

Ví dụ:

```python
from functools import partial
from src.utils import accuracy_under_attack

clean = accuracy_under_attack(model, x, y)
acc_fgsm = accuracy_under_attack(model, x, y, lambda m, xb, yb: fgsm(m, xb, yb, 8/255))
acc_pgd  = accuracy_under_attack(model, x, y, partial(pgd, eps=8/255, alpha=2/255, steps=10))
```

## 3. Tập đánh giá chung

```python
from src.data import get_eval_subset
x_eval, y_eval = get_eval_subset(as_tensors=True)   # 1000 ảnh test cố định, seed 42
```

Mọi con số accuracy dưới tấn công trong báo cáo đều đo trên tập này (trừ khi ghi rõ là toàn bộ test set).

## 4. Checkpoint

Tất cả là `state_dict` của `build_model()`, nằm trong `config.CHECKPOINT_DIR` (trên Colab: `MyDrive/AdvML_Project/checkpoints`):

| File | Nội dung | Tên trong config |
|---|---|---|
| `baseline_resnet18.pth` | Mô hình huấn luyện trên dữ liệu sạch | `config.BASELINE_CKPT` |
| `adv_fast_resnet18.pth` | Adversarial training kiểu Fast-AT | `config.ADV_CKPT_FAST` |
| `adv_pgd_resnet18.pth` | Adversarial training kiểu PGD-AT | `config.ADV_CKPT_PGD` |

Ví dụ:

```python
from src.models import build_model
from src.utils import load_checkpoint

model = build_model().to(device)
load_checkpoint(config.CHECKPOINT_DIR / config.ADV_CKPT_FAST, model, map_location=device)
model.eval()
```

## 5. File kết quả `results/*.json`

Các khóa chung:

- `"model"`: tên mô hình/cấu hình.
- `"clean_acc"`: accuracy trên ảnh sạch (%).
- `"fgsm"`: `{"1/255": acc, "2/255": acc, ...}`
- `"pgd"`: `{"1/255": acc, "2/255": acc, ...}`

Khóa eps dạng chuỗi `"k/255"`. Mỗi file **được thêm khóa riêng** tùy notebook.

Ví dụ (số liệu chỉ minh họa):

```json
{
  "model": "baseline_resnet18",
  "clean_acc": 94.1,
  "fgsm": {"1/255": 70.2, "2/255": 50.3, "4/255": 30.1, "8/255": 15.0},
  "pgd":  {"1/255": 60.4, "2/255": 25.7, "4/255": 3.2,  "8/255": 0.0}
}
```

Chuyển eps sang khóa: `f"{round(eps * 255)}/255"`.
