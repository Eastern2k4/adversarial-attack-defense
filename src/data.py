"""DataLoader cho CIFAR-10. Ảnh trả về nằm trong [0, 1] — KHÔNG Normalize ở đây
(Normalize nằm bên trong mô hình, xem src/models.py)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_DIR, NUM_WORKERS, N_EVAL, SEED

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])


def get_datasets(data_dir=DATA_DIR, download=True):
    train_set = datasets.CIFAR10(str(data_dir), train=True, download=download, transform=TRAIN_TRANSFORM)
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=download, transform=TEST_TRANSFORM)
    return train_set, test_set


def get_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, data_dir=DATA_DIR, download=True):
    """Trả về (train_loader, test_loader)."""
    train_set, test_set = get_datasets(data_dir, download)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def get_eval_indices(n=N_EVAL, seed=SEED, total=10000):
    """n chỉ số ảnh test cố định (cùng seed -> cùng tập ảnh trên mọi máy)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=g)[:n].tolist()


def get_eval_subset(n=N_EVAL, seed=SEED, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    data_dir=DATA_DIR, download=True, as_tensors=False):
    """Lấy n ảnh test cố định theo seed để đánh giá tấn công/phòng thủ.

    - as_tensors=False: trả về DataLoader (không shuffle).
    - as_tensors=True : trả về (images [n,3,32,32] trong [0,1], labels [n]).
    """
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=download, transform=TEST_TRANSFORM)
    subset = Subset(test_set, get_eval_indices(n, seed, len(test_set)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if not as_tensors:
        return loader
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)
