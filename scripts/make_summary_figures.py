"""Tạo biểu đồ và bảng tổng hợp từ các file kết quả đã có trong results/ (không cần GPU, checkpoint hay dữ liệu).

Sinh ra:
- results/adv_fast_vs_baseline_eps.png   (cùng cách vẽ với cell 16 của 03a_adversarial_training.ipynb)
- results/preprocessing_accuracy.png     (cùng cách vẽ với cell 15 của 03b_preprocessing.ipynb)
- results/summary_robustness_eps.png     (baseline / Fast-AT / PGD-AT theo ε, FGSM và PGD-10)
- results/summary_tradeoff.png           (đánh đổi accuracy sạch ↔ accuracy dưới tấn công, mọi phương pháp)
- results/summary_comparison.csv / .md   (bảng so sánh tổng hợp; thêm cột chi phí suy luận nếu có
                                          results/inference_cost.json từ scripts/measure_inference_cost.py)

Chạy từ thư mục gốc repo:  python scripts/make_summary_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
EPS_KEYS = ["1/255", "2/255", "4/255", "8/255"]
EPS_X = [0, 1, 2, 4, 8]

# Bảng màu (danh tính cố định theo phương pháp, dùng chung cho mọi biểu đồ tổng hợp)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {
    "Baseline": "#52514e",   # tham chiếu: xám trung tính
    "Fast-AT": "#2a78d6",    # slot 1 – xanh dương
    "PGD-AT": "#eb6834",     # slot 2 – cam
    "Preprocessing": "#1baf7a",  # slot 3 – xanh ngọc
}


def load(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. Hai biểu đồ còn thiếu, vẽ đúng như notebook gốc
# ---------------------------------------------------------------------------
def notebook_eps_plot(adv, mode):
    """Tái tạo cell 16 của 03a: nét liền = FGSM, nét đứt = PGD-10."""
    base = adv["baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for res, label, color in [(adv, f"AT-{mode}", "tab:blue"), (base, "Baseline", "tab:red")]:
        clean = res["clean_acc_eval_subset"]
        ax.plot(EPS_X, [clean] + [res["fgsm"][k] for k in EPS_KEYS], "-o", color=color, label=f"{label} – FGSM")
        ax.plot(EPS_X, [clean] + [res["pgd"][k] for k in EPS_KEYS], "--s", color=color, label=f"{label} – PGD-10")
    ax.set_xticks(EPS_X)
    ax.set_xticklabels(["0"] + [f"{e}/255" for e in EPS_X[1:]])
    ax.set_xlabel("ε (L∞)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Accuracy theo ε – baseline vs adversarial training ({mode})")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    out = RES / f"adv_{mode}_vs_baseline_eps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


PRE_LABEL = {"none": "Không xử lý", "jpeg": "JPEG q={}", "bit_depth": "{}-bit", "blur": "Blur σ={}"}


def pre_label(r):
    return PRE_LABEL[r["transform"]].format(r["param"])


def notebook_preprocessing_plot(pre):
    """Tái tạo cell 15 của 03b (biểu đồ cột nhóm, màu mặc định giống pandas)."""
    cols = [("Sạch", lambda r: r["clean_acc"]),
            ("FGSM (không thích nghi)", lambda r: r["fgsm"]["8/255"]),
            ("PGD (không thích nghi)", lambda r: r["pgd"]["8/255"]),
            ("PGD-BPDA (thích nghi)", lambda r: r["pgd_bpda_8/255"])]
    rows = list(pre.values())
    n, k, width = len(rows), len(cols), 0.8
    fig, ax = plt.subplots(figsize=(13, 5))
    for j, (name, get) in enumerate(cols):
        xs = [i - width / 2 + (j + 0.5) * width / k for i in range(n)]
        ax.bar(xs, [get(r) for r in rows], width / k, label=name, color=f"C{j}")
    ax.set_xticks(range(n))
    ax.set_xticklabels([pre_label(r) for r in rows], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_xlabel(""); ax.set_ylim(0, 100)
    ax.set_title("Tiền xử lý đầu vào trên baseline – ε = 8/255")
    ax.grid(axis="y", alpha=.3); ax.legend(loc="upper right")
    plt.tight_layout()
    out = RES / "preprocessing_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Biểu đồ tổng hợp
# ---------------------------------------------------------------------------
def summary_eps_plot(models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (atk, title) in zip(axes, [("fgsm", "FGSM"), ("pgd", "PGD-10 (α = 2/255, random start)")]):
        ends = []
        for name, res in models.items():
            ys = [res["clean_acc_eval_subset"]] + [res[atk][k] for k in EPS_KEYS]
            ax.plot(EPS_X, ys, "-o", color=COLOR[name], linewidth=2, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=name, zorder=3)
            ends.append([ys[-1], ys[-1], name])
        # Nhãn cuối đường: giãn cách tối thiểu 5 điểm % để không chồng chữ
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + 5)
        for val, ypos, name in ends:
            ax.annotate(f"{name} {val:.1f}%", (EPS_X[-1], val), xytext=(EPS_X[-1] + 0.25, ypos),
                        textcoords="data", va="center", fontsize=9, color=INK)
        ax.set_xticks(EPS_X)
        ax.set_xticklabels(["0"] + [f"{e}/255" for e in EPS_X[1:]])
        ax.set_xlim(-0.3, 10.8)
        ax.set_ylim(0, 100)
        ax.set_xlabel("ε (L∞)", color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("Accuracy (%) – 1.000 ảnh test", color=INK2)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Độ bền vững theo ε: baseline so với adversarial training", x=0.01, ha="left",
                 fontsize=12, color=INK)
    plt.tight_layout()
    out = RES / "summary_robustness_eps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def summary_tradeoff_plot(models, pre):
    """Hai khung: trái = toàn cảnh; phải = phóng to vùng các cấu hình tiền xử lý dồn sát đáy."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    c = COLOR["Preprocessing"]
    ZOOM = (82.5, 95.5, -1.2, 5.0)  # x0, x1, y0, y1 của vùng phóng to
    fig, (ax, az) = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.45, 1]})

    def draw_pre(a, big):
        # rỗng (lớn) = PGD chuyển từ baseline; đặc (nhỏ, nằm trong vòng rỗng nếu trùng) = PGD-BPDA
        for key, r in pre.items():
            if key == "none":
                continue
            x, yn, yb = r["clean_acc"], r["pgd"]["8/255"], r["pgd_bpda_8/255"]
            a.plot([x, x], [yn, yb], color=c, linewidth=1.2, alpha=.6, zorder=2)
            a.scatter([x], [yn], s=110 if big else 70, facecolor="white", edgecolor=c, linewidth=2, zorder=3)
            a.scatter([x], [yb], s=34 if big else 30, color=c, zorder=4)

    def draw_models(a):
        for name, res in models.items():
            a.scatter([res["clean_acc_eval_subset"]], [res["pgd20"]["8/255"]], s=120, marker="D",
                      color=COLOR[name], edgecolor="white", linewidth=1.5, zorder=5)

    # ---- Khung trái: toàn cảnh
    draw_pre(ax, big=False)
    draw_models(ax)
    for key in ("jpeg_50", "jpeg_75", "blur_1.0"):
        r = pre[key]
        ax.annotate(pre_label(r), (r["clean_acc"], r["pgd"]["8/255"]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    label_pos = {"Fast-AT": ((10, -4), "left"), "PGD-AT": ((-4, 12), "left")}
    for name, (offset, ha) in label_pos.items():
        res = models[name]
        x, y = res["clean_acc_eval_subset"], res["pgd20"]["8/255"]
        ax.annotate(f"{name}\nsạch {x:.1f}% · PGD-20 {y:.1f}%", (x, y), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK, va="bottom", ha=ha)
    x0, x1, y0, y1 = ZOOM
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK2, lw=1, ls="--", zorder=6))
    ax.annotate("phóng to ở khung phải →\n(Baseline và 6 điểm sát đáy)", (x1, y1), xytext=(-6, 34),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    handles = [
        Line2D([], [], marker="D", ls="", color=COLOR["Baseline"], markersize=8, label="Baseline (PGD-20)"),
        Line2D([], [], marker="D", ls="", color=COLOR["Fast-AT"], markersize=8, label="Fast-AT (PGD-20)"),
        Line2D([], [], marker="D", ls="", color=COLOR["PGD-AT"], markersize=8, label="PGD-AT (PGD-20)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=c, markeredgewidth=2,
               markersize=8, label="Tiền xử lý – kẻ tấn công không biết phòng thủ"),
        Line2D([], [], marker="o", ls="", color=c, markersize=6, label="Tiền xử lý – kẻ tấn công thích nghi (BPDA)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(25, 100); ax.set_ylim(-2, 70)
    ax.set_xlabel("Accuracy trên ảnh sạch (%)", color=INK2)
    ax.set_ylabel("Accuracy dưới tấn công, ε = 8/255 (%)", color=INK2)
    ax.set_title("Toàn cảnh", loc="left", fontsize=11, color=INK)
    style_axes(ax)

    # ---- Khung phải: phóng to, gắn nhãn từng điểm bằng đường dẫn
    draw_pre(az, big=True)
    draw_models(az)
    b = models["Baseline"]
    # (khóa, vị trí nhãn x, y) – đặt nhãn so le để không chồng nhau
    labels = [
        ("jpeg_75", 83.0, 4.55, "JPEG q=75 (BPDA 3,3%;\nkhông biết: 43,2% – ngoài khung)"),
        ("bit_depth_3", 85.6, 2.35, None),
        ("jpeg_90", 88.2, 3.15, None),
        ("blur_0.5", 90.0, 4.1, None),
        ("bit_depth_4", 89.2, -0.85, None),
        ("bit_depth_5", 92.1, 1.75, None),
    ]
    for key, lx, ly, text in labels:
        r = pre[key]
        x, yn, yb = r["clean_acc"], r["pgd"]["8/255"], r["pgd_bpda_8/255"]
        txt = text or f"{pre_label(r)}  ({yn:.1f}% → {yb:.1f}%)".replace(".", ",")
        tgt_y = yb if key == "jpeg_75" else yn
        az.annotate(txt, (x, tgt_y), xytext=(lx, ly), textcoords="data", fontsize=8.5, color=INK,
                    ha="left", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=6))
    az.annotate("Baseline (94,0% · 0,0%)", (b["clean_acc_eval_subset"], b["pgd20"]["8/255"]),
                xytext=(93.0, 3.0), textcoords="data", fontsize=8.5, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=7))
    az.set_xlim(x0, x1); az.set_ylim(y0, y1); az.set_yticks(range(0, 6))
    az.set_xlabel("Accuracy trên ảnh sạch (%)", color=INK2)
    az.set_title("Phóng to vùng sát đáy · nhãn: không biết → BPDA", loc="left", fontsize=11, color=INK)
    style_axes(az)
    for side in ("left", "bottom", "top", "right"):
        az.spines[side].set_visible(True); az.spines[side].set_color(INK2); az.spines[side].set_linestyle("--")

    fig.suptitle("Đánh đổi: accuracy sạch và độ bền vững (ε = 8/255, 1.000 ảnh test)", x=0.01, ha="left",
                 fontsize=12.5, color=INK)
    plt.tight_layout()
    out = RES / "summary_tradeoff.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Bảng so sánh tổng hợp
# ---------------------------------------------------------------------------
def fmt(v, nd=1):
    return "–" if v is None else f"{v:.{nd}f}"


def summary_table(models, fast, pgd_at, pre, infer):
    train = {"Baseline": None,
             "Fast-AT": fast["train_time_per_epoch_sec"],
             "PGD-AT": pgd_at["train_time_per_epoch_sec"]}
    rows = []
    for name, res in models.items():
        rows.append({
            "Phương pháp": name,
            "Sạch (10k test)": res["clean_acc"],
            "Sạch (1k)": res["clean_acc_eval_subset"],
            "FGSM 8/255": res["fgsm"]["8/255"],
            "PGD-10 8/255": res["pgd"]["8/255"],
            "PGD-20 8/255": res["pgd20"]["8/255"],
            "Tấn công thích nghi 8/255": res["pgd20"]["8/255"],
            "Huấn luyện (giây/epoch)": train[name],
            "Suy luận (ms/ảnh)": (infer or {}).get(name),
        })
    for key, r in pre.items():
        if key == "none":
            continue
        rows.append({
            "Phương pháp": f"Baseline + {pre_label(r)}",
            "Sạch (10k test)": None,
            "Sạch (1k)": r["clean_acc"],
            "FGSM 8/255": r["fgsm"]["8/255"],
            "PGD-10 8/255": r["pgd"]["8/255"],
            "PGD-20 8/255": None,
            "Tấn công thích nghi 8/255": r["pgd_bpda_8/255"],
            "Huấn luyện (giây/epoch)": 0.0,
            "Suy luận (ms/ảnh)": (infer or {}).get(key),
        })
    cols = list(rows[0].keys())
    with open(RES / "summary_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [r["Phương pháp"]]
        for c in cols[1:]:
            nd = 2 if c == "Suy luận (ms/ảnh)" else (0 if c == "Huấn luyện (giây/epoch)" else 1)
            cells.append(fmt(r[c], nd))
        lines.append("| " + " | ".join(cells) + " |")
    notes = [
        "",
        "Ghi chú:",
        "- Accuracy dưới tấn công đo trên 1.000 ảnh test cố định (seed 42), ε theo chuẩn L∞, white-box, untargeted.",
        "- Baseline/AT: FGSM, PGD-10 và PGD-20 tấn công trực tiếp mô hình, nên PGD-20 cũng là tấn công thích nghi.",
        "- Tiền xử lý: FGSM/PGD-10 sinh trên baseline rồi mới qua bước tiền xử lý (kẻ tấn công không biết phòng thủ);"
        " cột 'thích nghi' là PGD-10 + BPDA tấn công trực tiếp f(g(x)).",
        "- Số liệu baseline lấy từ cùng lượt chạy đánh giá với AT (adv_training_*_metrics.json) để so sánh cùng"
        " điều kiện; lượt đo riêng của phần tấn công (attack_metrics.json, CPU) lệch ≤ 0,1 điểm.",
        "- Thời gian huấn luyện đo trên GPU Colab; baseline không ghi thời gian mỗi epoch. Tiền xử lý không cần huấn luyện lại.",
    ]
    if infer:
        notes.append(f"- Suy luận: {infer.get('_note', '')}")
    else:
        notes.append("- Chưa có results/inference_cost.json: chạy scripts/measure_inference_cost.py để điền cột suy luận.")
    with open(RES / "summary_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines + notes) + "\n")
    return RES / "summary_comparison.md"


def main():
    fast = load("adv_training_fast_metrics.json")
    pgd_at = load("adv_training_pgd_metrics.json")
    pre = load("preprocessing_metrics.json")
    infer_path = RES / "inference_cost.json"
    infer = None
    if infer_path.exists():
        data = json.loads(infer_path.read_text(encoding="utf-8"))
        infer = {k: v["ms_per_image"] for k, v in data["results"].items()}
        infer["_note"] = data.get("note", "")
    models = {"Baseline": fast["baseline"], "Fast-AT": fast, "PGD-AT": pgd_at}
    outs = [
        notebook_eps_plot(fast, "fast"),
        notebook_preprocessing_plot(pre),
        summary_eps_plot(models),
        summary_tradeoff_plot(models, pre),
        summary_table(models, fast, pgd_at, pre, infer),
    ]
    for o in outs:
        print("Đã lưu", o.relative_to(ROOT))


if __name__ == "__main__":
    main()
