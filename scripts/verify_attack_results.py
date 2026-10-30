"""Đối chiếu kết quả tấn công đã lưu; không chạy lại benchmark hay huấn luyện."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.attacks import fgsm, pgd  # noqa: E402
from src.data import get_eval_indices  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(checkpoint=None, results_dir=None):
    results = Path(results_dir) if results_dir is not None else ROOT / "results"
    metrics = json.loads((results / "attack_metrics.json").read_text(encoding="utf-8"))
    with (results / "attack_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with (results / "attack_metrics.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))

    require(str(inspect.signature(fgsm)) == "(model, x, y, eps, targeted=False)", "FGSM signature changed")
    require(
        str(inspect.signature(pgd))
        == "(model, x, y, eps, alpha, steps, random_start=True, targeted=False)",
        "PGD signature changed",
    )
    require(metrics["evaluation"]["completed"] is True, "Evaluation is incomplete")
    require(metrics["model"] == Path(config.BASELINE_CKPT).stem, "Unexpected model")
    require(metrics["evaluation"]["n_eval"] == len(predictions) == config.N_EVAL, "Wrong evaluation size")
    require(metrics["evaluation"]["seed"] == config.SEED, "Wrong seed")
    indices = [int(row["test_index"]) for row in predictions]
    require(indices == get_eval_indices(), "Test indices differ from the shared seed subset")
    index_hash = hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    require(index_hash == metrics["evaluation"]["eval_indices_sha256"], "Test index hash mismatch")

    clean_correct = [int(row["clean_pred"]) == int(row["true_label"]) for row in predictions]
    clean_count = sum(clean_correct)
    require(clean_count == metrics["evaluation"]["clean_correct"], "Clean count mismatch")
    require(metrics["clean_acc"] == 100.0 * clean_count / len(predictions), "Clean accuracy mismatch")
    expected_keys = [f"{round(eps * 255)}/255" for eps in config.EPSILONS]
    require(len(table) == 1 + 2 * len(expected_keys), "CSV row count mismatch")
    clean_rows = [row for row in table if row["attack"] == "clean"]
    require(len(clean_rows) == 1, "Missing or duplicate clean row")
    require(float(clean_rows[0]["accuracy_pct"]) == metrics["clean_acc"], "Clean CSV mismatch")
    attack_config = metrics["attack_config"]
    require(attack_config["epsilons"] == expected_keys, "Epsilon configuration mismatch")
    require(attack_config["pgd_steps"] == config.PGD_STEPS, "PGD steps mismatch")
    require(attack_config["pgd_alpha"] == config.PGD_ALPHA, "PGD alpha mismatch")
    require(attack_config["norm"] == "Linf" and attack_config["targeted"] is False, "Wrong attack scenario")
    require(attack_config["random_start"] is True and attack_config["restarts"] == 1, "Wrong PGD initialization")
    require(attack_config["seed_per_attack"] == config.SEED, "Wrong attack RNG seed")

    for attack in ("fgsm", "pgd"):
        require(list(metrics[attack]) == expected_keys, f"Missing epsilon results: {attack}")
        for eps, key in zip(config.EPSILONS, expected_keys):
            correct = [int(row[f"{attack}_{key}"]) == int(row["true_label"]) for row in predictions]
            successes = sum(clean and not adv for clean, adv in zip(clean_correct, correct))
            accuracy = 100.0 * sum(correct) / len(predictions)
            asr = 100.0 * successes / clean_count if clean_count else None
            require(metrics[attack][key] == accuracy, f"Accuracy mismatch: {attack} {key}")
            require(metrics["asr"][attack][key] == asr, f"ASR mismatch: {attack} {key}")
            detail = metrics["details"][attack][key]
            require(detail["correct"] == sum(correct), f"Correct count mismatch: {attack} {key}")
            require(detail["successful_attacks_on_clean_correct"] == successes, f"ASR count mismatch: {attack} {key}")
            require(0 <= detail["max_linf"] <= eps + 1e-6, f"Linf violation: {attack} {key}")
            require(0 <= detail["pixel_min"] <= detail["pixel_max"] <= 1, f"Pixel range violation: {attack} {key}")
            matching = [row for row in table if row["attack"] == attack and row["epsilon"] == key]
            require(len(matching) == 1, f"Missing or duplicate CSV result: {attack} {key}")
            require(float(matching[0]["accuracy_pct"]) == accuracy, f"CSV accuracy mismatch: {attack} {key}")
            csv_asr = float(matching[0]["asr_pct"]) if matching[0]["asr_pct"] else None
            require(csv_asr == asr, f"CSV ASR mismatch: {attack} {key}")

    source = (ROOT / "src/attacks.py").read_bytes()
    lf_source = source.replace(b"\r\n", b"\n")
    # Git có thể đổi LF/CRLF khi checkout trên máy khác; nội dung mã vẫn giống nhau.
    source_hashes = {
        hashlib.sha256(content).hexdigest()
        for content in (source, lf_source, lf_source.replace(b"\n", b"\r\n"))
    }
    require(metrics["provenance"]["attacks_py_sha256"] in source_hashes, "Attack source differs from the measured version")
    notebook = nbformat.read(ROOT / "notebooks/02_attacks.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            require(cell.execution_count is None and cell.outputs == [], "Clear notebook outputs before committing")
            compile(cell.source, "02_attacks.ipynb", "exec")
    for name in ("attack_accuracy.png", "attack_examples_fgsm.png", "attack_examples_pgd.png"):
        with Image.open(results / name) as picture:
            picture.verify()
    if checkpoint is not None:
        require(sha256_file(checkpoint) == metrics["provenance"]["checkpoint_sha256"], "Checkpoint hash mismatch")

    print(f"PASS: {len(predictions)} images, shared seed {config.SEED}, clean accuracy {metrics['clean_acc']:.1f}%")
    print("PASS: public signatures, JSON/CSV accuracy and ASR, recorded noise bounds, source hash, notebook and PNGs")
    if checkpoint is not None:
        print("PASS: checkpoint SHA-256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional baseline checkpoint to verify by SHA-256")
    parser.add_argument("--results-dir", type=Path, help="Folder containing the attack results downloaded from Drive")
    args = parser.parse_args()
    try:
        verify(args.checkpoint, args.results_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
