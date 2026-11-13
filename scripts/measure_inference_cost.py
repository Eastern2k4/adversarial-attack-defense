"""Đo chi phí suy luận (ms/ảnh) của mô hình gốc và của từng phép tiền xử lý.

Thời gian suy luận chỉ phụ thuộc kiến trúc và phép tiền xử lý, không phụ thuộc trọng số:
baseline, Fast-AT và PGD-AT cùng là ResNet-18 nên có cùng chi phí suy luận (AT chỉ tốn thêm lúc huấn luyện).
Nếu có checkpoint baseline thì nạp vào; không có thì dùng trọng số khởi tạo (thời gian như nhau).

Chạy từ thư mục gốc repo:  python scripts/measure_inference_cost.py [--threads 4] [--n 1000] [--repeats 7]
Ghi ra results/inference_cost.json; sau đó chạy lại scripts/make_summary_figures.py để điền vào bảng tổng hợp.
"""
import argparse
import json
import platform
import statistics
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src import config  # noqa: E402
from src.data import get_eval_subset  # noqa: E402
from src.defenses import PreprocessedModel, gaussian_blur, jpeg_compress, reduce_bit_depth  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_checkpoint  # noqa: E402


@torch.no_grad()
def time_once(model, x, batch_size):
    t0 = time.perf_counter()
    for i in range(0, len(x), batch_size):
        model(x[i:i + batch_size])
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=config.N_EVAL)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model = build_model().to(device)
    ckpt = config.CHECKPOINT_DIR / config.BASELINE_CKPT
    weights = "baseline_resnet18.pth" if ckpt.exists() else "khởi tạo ngẫu nhiên (không ảnh hưởng thời gian)"
    if ckpt.exists():
        load_checkpoint(ckpt, model, map_location=device)

    x, _ = get_eval_subset(n=args.n, as_tensors=True)

    transforms = {"Baseline": None}
    transforms.update({f"jpeg_{q}": partial(jpeg_compress, quality=q) for q in config.JPEG_QUALITIES})
    transforms.update({f"bit_depth_{b}": partial(reduce_bit_depth, bits=b) for b in config.BIT_DEPTHS})
    transforms.update({f"blur_{s}": partial(gaussian_blur, sigma=s) for s in config.BLUR_SIGMAS})

    models = {key: (model if g is None else PreprocessedModel(model, g)).eval() for key, g in transforms.items()}
    for m in models.values():  # warm-up
        time_once(m, x[:args.batch_size], args.batch_size)
    # Đo luân phiên giữa các phương pháp để nhiễu do xung nhịp CPU chia đều, lấy trung vị
    runs = {key: [] for key in models}
    for _ in range(args.repeats):
        for key, m in models.items():
            runs[key].append(time_once(m, x, args.batch_size))
    results = {}
    for key, ts in runs.items():
        ms = statistics.median(ts) * 1000 / len(x)
        results[key] = {"ms_per_image": ms}
        print(f"{key:<14s} {ms:7.3f} ms/ảnh")
    base = results["Baseline"]["ms_per_image"]
    for key, r in results.items():
        r["overhead_vs_baseline_pct"] = (r["ms_per_image"] / base - 1) * 100
    # AT không đổi kiến trúc -> cùng chi phí suy luận với baseline
    results["Fast-AT"] = dict(results["Baseline"])
    results["PGD-AT"] = dict(results["Baseline"])

    cpu = platform.processor() or platform.machine()
    note = (f"CPU ({cpu}), {args.threads} luồng, batch {args.batch_size}, {args.n} ảnh, trung vị {args.repeats} lần đo luân phiên; "
            f"Fast-AT/PGD-AT cùng kiến trúc với baseline nên cùng chi phí suy luận.")
    out = {
        "note": note,
        "weights": weights,
        "device": "cpu",
        "threads": args.threads,
        "batch_size": args.batch_size,
        "n_images": args.n,
        "repeats": args.repeats,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "results": results,
    }
    path = config.RESULTS_DIR / "inference_cost.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã lưu", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
