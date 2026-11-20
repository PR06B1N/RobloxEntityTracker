"""Stage 4 - train the `character` detector from YOUR labelled dataset.

    python train.py                      # defaults: yolo11n.pt, 100 epochs, imgsz 640
    python train.py --epochs 150 --model yolo11s.pt

The pretrained starting weights (COCO) are downloaded automatically by
Ultralytics on first use - that is *transfer learning*; the Roblox knowledge
comes entirely from your dataset. The best checkpoint is copied to
models/best.pt, together with models/training_info.json that records the real
metrics measured on your validation / test images.

This script refuses to run if there is no labelled dataset.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from config import DATA_YAML, DATASET_DIR, MODELS_DIR, ROOT
from dataset_tools import DatasetError, print_report, validate_dataset

MIN_TRAIN_IMAGES = 50       # below this training is pointless -> hard stop
RECOMMENDED_IMAGES = 500    # project target for a first usable model


def resolve_device(requested: str) -> str:
    """'auto' -> first CUDA GPU if PyTorch can use it, otherwise 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def check_dataset(dataset_dir: Path = DATASET_DIR, min_train: int = MIN_TRAIN_IMAGES) -> dict:
    """Raise DatasetError unless the dataset is usable. Returns per-split image counts."""
    reports = validate_dataset(dataset_dir)
    counts = {split: r.images for split, r in reports.items()}
    if counts["train"] == 0 and counts["val"] == 0:
        raise DatasetError(
            "No dataset found. Collect screenshots (collect_dataset.py), annotate them in CVAT, "
            "then run: python dataset_tools.py import-cvat <export.zip>"
        )
    if counts["train"] < min_train:
        raise DatasetError(f"Only {counts['train']} training images (minimum {min_train}). Label more screenshots first.")
    if counts["val"] == 0:
        raise DatasetError("The validation split is empty. Re-run import-cvat with --val 0.2.")
    labelled = sum(r.boxes for r in reports.values())
    if labelled == 0:
        raise DatasetError("The dataset has images but no boxes at all.")
    broken = [i for r in reports.values() for i in r.issues if "no label file" in i or "label without image" in i]
    if broken:
        raise DatasetError("Dataset is inconsistent (e.g. " + broken[0] + "). Run: python dataset_tools.py stats")
    return counts


def build_runtime_yaml(data_yaml: Path, out_path: Path, root: Optional[Path] = None) -> Path:
    """Copy data.yaml with an absolute `path:` so it works no matter where we run from.

    A relative `path:` is resolved against the folder that contains data.yaml.
    """
    data_yaml = Path(data_yaml)
    root = Path(root) if root else data_yaml.resolve().parent
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", "dataset"))
    data["path"] = str((base if base.is_absolute() else root / base).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def _metric_dict(metrics) -> dict:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }


def train(args: argparse.Namespace) -> Path:
    counts = check_dataset(Path(args.dataset), args.min_train_images)
    print(f"Dataset images: {counts}")
    if counts["train"] + counts["val"] + counts["test"] < RECOMMENDED_IMAGES:
        print(f"WARNING: fewer than {RECOMMENDED_IMAGES} images. Training will run, but expect a weak detector.")

    from ultralytics import YOLO  # imported late: it is slow and needs PyTorch
    import ultralytics

    device = resolve_device(args.device)
    batch = args.batch if args.batch else (16 if device != "cpu" else 8)
    runs_dir = Path(args.runs_dir).resolve()
    runtime_yaml = build_runtime_yaml(Path(args.data), runs_dir / "runtime_data.yaml")
    print(f"Device: {device} | batch: {batch} | start weights: {args.model}")

    model = YOLO(args.model)  # pretrained weights -> transfer learning
    results = model.train(
        data=str(runtime_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        patience=args.patience,
        project=str(runs_dir),
        name=args.name,
        exist_ok=False,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        flipud=0.0,        # characters are never upside down
        plots=True,
    )
    save_dir = Path(results.save_dir) if getattr(results, "save_dir", None) else Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but {best} was not produced.")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / "best.pt"
    target.write_bytes(best.read_bytes())

    # Real numbers measured on held-out images, from the best checkpoint.
    best_model = YOLO(str(target))
    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "start_weights": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "device": device,
        "dataset_images": counts,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "validation": _metric_dict(best_model.val(data=str(runtime_yaml), split="val", device=device, plots=False, verbose=False,
                                          project=str(runs_dir), name=f"{args.name}_val", exist_ok=True)),
    }
    if counts["test"] > 0:
        info["test"] = _metric_dict(best_model.val(data=str(runtime_yaml), split="test", device=device, plots=False, verbose=False,
                                           project=str(runs_dir), name=f"{args.name}_test", exist_ok=True))
    (export_dir / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved {target}")
    print(json.dumps({k: info[k] for k in ("validation", "test") if k in info}, indent=2))
    print("These numbers come from your own validation/test images. Check the overlay on real gameplay too.")
    return target


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Roblox 'character' YOLO detector.")
    p.add_argument("--model", default="yolo11n.pt", help="pretrained starting weights (yolo11n.pt = fastest, yolo11s.pt = more accurate)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0 = automatic (16 on GPU, 8 on CPU)")
    p.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1 ...")
    p.add_argument("--patience", type=int, default=30, help="stop early after this many epochs without improvement")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers (use 0 if Windows complains)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=datetime.now().strftime("character_%Y%m%d_%H%M%S"))
    p.add_argument("--data", default=str(DATA_YAML))
    p.add_argument("--dataset", default=str(DATASET_DIR))
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--export-dir", default=str(MODELS_DIR), help="where best.pt is copied")
    p.add_argument("--min-train-images", type=int, default=MIN_TRAIN_IMAGES)
    p.add_argument("--check-only", action="store_true", help="only validate the dataset, do not train")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.check_only:
            counts = check_dataset(Path(args.dataset), args.min_train_images)
            print_report(validate_dataset(Path(args.dataset)))
            print(f"Dataset OK for training: {counts}")
            return 0
        train(args)
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
