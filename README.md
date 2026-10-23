# Đề tài 2.21 – Nghiên cứu và phòng thủ trước tấn công đối kháng lên mô hình phân loại ảnh

Đồ án môn **An toàn thông tin**. Mô hình ResNet-18 (bản cho CIFAR) huấn luyện trên CIFAR-10, đánh giá trước các tấn công đối kháng FGSM/PGD và các biện pháp phòng thủ Adversarial Training, Input Preprocessing.

## Thành viên

| Thành viên | Phụ trách |
|---|---|
| Nguyễn Quang Huy | Tấn công (FGSM, PGD) – `notebooks/02_attacks.ipynb` |
| Nguyễn Phương Đông | Phòng thủ (Adversarial Training, Input Preprocessing) – `notebooks/03a_adversarial_training.ipynb`, `notebooks/03b_preprocessing.ipynb` |

## Cấu trúc thư mục

```
├── notebooks/
│   ├── 01_baseline.ipynb    # Huấn luyện & đánh giá mô hình baseline
│   ├── 02_attacks.ipynb     # Tấn công FGSM, PGD
│   ├── 03a_adversarial_training.ipynb  # Phòng thủ: Fast-AT / PGD-AT
│   └── 03b_preprocessing.ipynb         # Phòng thủ: JPEG, giảm bit, blur (+ tấn công thích nghi BPDA)
├── src/
│   ├── config.py            # Toàn bộ thông số dùng chung
│   ├── models.py            # Normalize + ResNet-18 (CIFAR), build_model()
│   ├── data.py              # DataLoader CIFAR-10, tập ảnh đánh giá cố định
│   ├── attacks.py           # fgsm(), pgd()
│   ├── defenses.py          # Fast-AT, PGD-AT, tiền xử lý, PreprocessedModel (BPDA)
│   └── utils.py             # seed, train/evaluate, accuracy_under_attack, checkpoint
├── docs/
│   └── INTERFACE.md         # Giao diện đã chốt giữa phần tấn công và phòng thủ
├── results/                 # Hình, bảng số liệu (PNG/JSON) cho báo cáo
├── report/                  # Báo cáo
└── requirements.txt
```

**Quy ước quan trọng:** ảnh đưa vào mô hình luôn nằm trong khoảng **[0, 1]**. Chuẩn hoá mean/std được đặt **bên trong mô hình** (lớp `Normalize` đầu mạng), không nằm trong transform, để epsilon của tấn công có ý nghĩa đúng trên thang pixel (vd. ε = 8/255).

**Giao diện giữa phần tấn công và phòng thủ** (chữ ký `fgsm`/`pgd`, `accuracy_under_attack`, tập đánh giá, tên checkpoint, cấu trúc file `results/*.json`): xem [`docs/INTERFACE.md`](docs/INTERFACE.md).

## Thứ tự chạy notebook

| # | Notebook | Cần có trước | Tạo ra | Thời gian (T4) |
|---|---|---|---|---|
| 1 | `01_baseline.ipynb` | – | `baseline_resnet18.pth` | ~40 phút |
| 2 | `02_attacks.ipynb` | `baseline_resnet18.pth` | kết quả tấn công | – |
| 3a | `03a_adversarial_training.ipynb` (`MODE="fast"`) | – (so sánh với baseline nếu có) | `adv_fast_resnet18.pth`, `results/adv_training_*` | ~1 giờ |
| 3a | `03a_adversarial_training.ipynb` (`MODE="pgd"`) | – | `adv_pgd_resnet18.pth` | ~4–5 giờ (tự resume) |
| 3b | `03b_preprocessing.ipynb` | `baseline_resnet18.pth` | `results/preprocessing_*` | ~10 phút |

Notebook 3a không phụ thuộc baseline nên có thể chạy song song với Notebook 1 (trên tài khoản/phiên Colab khác).

## Chạy trên Google Colab (khuyến nghị)

1. Trong Google Drive, tạo thư mục `MyDrive/AdvML_Project` (notebook cũng tự tạo nếu chưa có). Checkpoint sẽ được lưu ở `MyDrive/AdvML_Project/checkpoints`.
2. Mở notebook từ GitHub: Colab → *File → Open notebook → GitHub* → dán `https://github.com/Eastern2k4/adversarial-attack-defense` → chọn notebook.
3. Bật GPU: *Runtime → Change runtime type → T4 GPU*.
4. Chạy lần lượt các cell. Cell setup sẽ mount Drive, clone repo vào `/content` và thêm vào `sys.path`.
5. Nếu runtime bị ngắt khi đang huấn luyện: kết nối lại và chạy lại từ đầu – quá trình huấn luyện tự resume từ checkpoint trên Drive.

> Thư mục `results/` trên Colab nằm trong `/content` nên sẽ mất khi runtime kết thúc – nhớ tải file kết quả về hoặc commit lên repo.

## Chạy local

```bash
git clone https://github.com/Eastern2k4/adversarial-attack-defense.git
cd adversarial-attack-defense
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Checkpoint lưu ở `./checkpoints`, dữ liệu CIFAR-10 tự tải về `./data`. Có GPU NVIDIA thì cài bản torch hỗ trợ CUDA theo hướng dẫn tại https://pytorch.org.

## Thông số thí nghiệm (`src/config.py`)

| Thông số | Giá trị |
|---|---|
| Dataset | CIFAR-10 (50.000 train / 10.000 test) |
| Mô hình | ResNet-18 bản CIFAR (conv đầu 3×3 stride 1, không maxpool) + Normalize |
| Seed | 42 |
| Batch size | 128 |
| Epochs | 40 |
| Optimizer | SGD, lr = 0.1, momentum = 0.9, weight decay = 5e-4 |
| LR scheduler | CosineAnnealingLR |
| Augmentation (train) | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| Epsilon (L∞) | 1/255, 2/255, 4/255, 8/255 |
| PGD | 10 bước, bước nhảy α = 2/255 |
| Số ảnh đánh giá tấn công | 1000 ảnh test cố định theo seed (`get_eval_subset`) |
| Adversarial training | 30 epoch, ε = 8/255; Fast-AT α = 10/255, OneCycleLR max_lr = 0.2; PGD-AT 7 bước α = 2/255 |
| Tiền xử lý | JPEG q ∈ {90, 75, 50}; giảm bit {5, 4, 3}; Gaussian blur σ ∈ {0.5, 1.0} |

## Quy ước làm việc nhóm

- Mỗi người **chỉ sửa notebook của mình** (A: `02_attacks.ipynb`, B: `03a_*.ipynb`, `03b_*.ipynb`). `src/attacks.py` hiện là bản tạm do B viết – A được viết lại nhưng giữ nguyên chữ ký hàm (`docs/INTERFACE.md`). Thay đổi trong `src/` cần báo cho người còn lại.
- Mọi thông số dùng chung đặt trong `src/config.py`, không hard-code trong notebook.
- **Clear all outputs** trước khi commit notebook (Colab: *Edit → Clear all outputs*; Jupyter: *Kernel → Restart & Clear Output*).
- **Không commit** dữ liệu (`data/`) và trọng số (`*.pth`, `*.pt`, `checkpoints/`). Trọng số chia sẻ qua thư mục Drive `AdvML_Project`.
- `git pull` trước khi bắt đầu làm việc để tránh xung đột.
