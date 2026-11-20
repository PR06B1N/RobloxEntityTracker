"""Stage 3b - turn a CVAT export into a clean train/val/test YOLO dataset.

Typical use (from the project folder):

    python dataset_tools.py import-cvat dataset/cvat_export/my_export.zip
    python dataset_tools.py stats

What `import-cvat` does
-----------------------
1. Finds every image and its YOLO label file inside the CVAT export
   (works with CVAT's "YOLO 1.1" and "Ultralytics YOLO Detection 1.0" formats).
2. Cleans the labels: single class `character` (id 0), values clipped to 0..1,
   broken lines dropped and reported.
3. Splits into train / val / test **by blocks of consecutive screenshots**, so
   near-identical neighbouring frames never end up on both sides of the split
   (that would make validation scores look better than reality).
4. Writes dataset/images/<split>, dataset/labels/<split> and data.yaml.
"""
from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import CLASS_NAMES, DATA_YAML, DATASET_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
_IGNORED_TXT = {"train.txt", "val.txt", "valid.txt", "test.txt", "classes.txt", "labels.txt", "obj.names"}


class DatasetError(Exception):
    """Raised for problems the user needs to fix (bad export, empty dataset, ...)."""


# --------------------------------------------------------------------------
# Label cleaning
# --------------------------------------------------------------------------
def clean_label_text(text: str) -> tuple[list[str], list[str]]:
    """Validate YOLO detection lines. Returns (clean_lines, warnings)."""
    clean: list[str] = []
    warnings: list[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"line {n}: expected 5 values, got {len(parts)} (polygon/segment?) - dropped")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            warnings.append(f"line {n}: non-numeric value - dropped")
            continue
        if cls != 0:
            warnings.append(f"line {n}: class id {cls} is not 0 ('character') - dropped")
            continue
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            warnings.append(f"line {n}: NaN/inf value - dropped")
            continue
        # clip the box to the image, then convert back to centre/size form
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            warnings.append(f"line {n}: empty or off-image box - dropped")
            continue
        if (x1, y1, x2, y2) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2):
            warnings.append(f"line {n}: box extended outside the image - clipped")
        clean.append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return clean, warnings


# --------------------------------------------------------------------------
# Finding images + labels inside an export
# --------------------------------------------------------------------------
def locate_pairs(root: Path) -> tuple[list[tuple[Path, Optional[Path]]], list[Path]]:
    """Match images to label files by file name. Returns (pairs, orphan_label_files)."""
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    labels = {p.stem: p for p in root.rglob("*.txt") if p.name.lower() not in _IGNORED_TXT}

    seen: dict[str, Path] = {}
    for img in images:
        if img.stem in seen:
            raise DatasetError(f"Two images share the name '{img.stem}': {seen[img.stem]} and {img}")
        seen[img.stem] = img

    pairs = [(img, labels.get(img.stem)) for img in images]
    orphans = [p for stem, p in labels.items() if stem not in seen]
    return pairs, orphans


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
_NAME_RE = re.compile(r"^(?P<session>.+)_(?P<num>\d+)$")


def group_key(stem: str, block: int) -> str:
    """'s1-session_00042' with block=25 -> 's1-session:1'. Unknown names are their own group."""
    m = _NAME_RE.match(stem)
    if not m:
        return stem
    return f"{m.group('session')}:{int(m.group('num')) // max(1, block)}"


def split_stems(stems: list[str], val_frac: float, test_frac: float,
                seed: int = 42, block: int = 25) -> dict[str, str]:
    """Assign each stem to 'train', 'val' or 'test' keeping blocks of frames together."""
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 0.9:
        raise DatasetError("val and test fractions must be >= 0 and add up to less than 0.9")
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        groups[group_key(stem, block)].append(stem)

    needed = 1 + (val_frac > 0) + (test_frac > 0)
    if len(groups) < needed:
        raise DatasetError(
            f"Only {len(groups)} block(s) of images found but {needed} are needed for the requested split. "
            f"Add more images or use a smaller --block (currently {block})."
        )

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = len(stems)
    target = {"test": round(test_frac * total), "val": round(val_frac * total)}
    counts = {"test": 0, "val": 0}
    assignment: dict[str, str] = {}
    for key in keys:
        if test_frac > 0 and counts["test"] < max(1, target["test"]):
            split = "test"
        elif val_frac > 0 and counts["val"] < max(1, target["val"]):
            split = "val"
        else:
            split = "train"
        if split != "train":
            counts[split] += len(groups[key])
        for stem in groups[key]:
            assignment[stem] = split
    if not any(s == "train" for s in assignment.values()):
        raise DatasetError("The split left no training images. Add more images or reduce --block.")
    return assignment


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------
@dataclass
class ImportSummary:
    images_found: int = 0
    imported: dict = field(default_factory=lambda: {s: 0 for s in SPLITS})
    boxes: dict = field(default_factory=lambda: {s: 0 for s in SPLITS})
    unlabeled_skipped: int = 0
    orphan_labels: int = 0
    warnings: list = field(default_factory=list)


def write_data_yaml(path: Path = DATA_YAML) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path.write_text(
        "# Ultralytics dataset description (one class: players and NPCs are both 'character').\n"
        "# Paths are relative to this file's folder; train.py converts them to absolute paths.\n"
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def import_cvat(export: Path, dest: Path = DATASET_DIR, val: float = 0.2, test: float = 0.1,
                seed: int = 42, block: int = 25, clean: bool = False,
                unlabeled_as_background: bool = False, yaml_path: Optional[Path] = DATA_YAML) -> ImportSummary:
    export = Path(export)
    if not export.exists():
        raise DatasetError(f"{export} does not exist.")

    with tempfile.TemporaryDirectory() as tmp:
        root = export
        if export.is_file():
            if export.suffix.lower() != ".zip":
                raise DatasetError("Give a CVAT export .zip file or an extracted folder.")
            with zipfile.ZipFile(export) as zf:
                zf.extractall(tmp)
            root = Path(tmp)

        pairs, orphans = locate_pairs(root)
        summary = ImportSummary(images_found=len(pairs), orphan_labels=len(orphans))
        if not pairs:
            raise DatasetError("No images found in the export. In CVAT enable 'Save images' when exporting.")

        usable: list[tuple[Path, list[str]]] = []
        for img, lab in pairs:
            if lab is None:
                if unlabeled_as_background:
                    usable.append((img, []))
                else:
                    summary.unlabeled_skipped += 1
                continue
            lines, warns = clean_label_text(lab.read_text(encoding="utf-8", errors="replace"))
            summary.warnings += [f"{lab.name}: {w}" for w in warns]
            usable.append((img, lines))
        if not usable:
            raise DatasetError("No image has a label file. Did you export in a YOLO format?")

        assignment = split_stems([img.stem for img, _ in usable], val, test, seed, block)

        out_images, out_labels = Path(dest) / "images", Path(dest) / "labels"
        if (out_images.exists() and any(out_images.rglob("*.*"))) or (out_labels.exists() and any(out_labels.rglob("*.txt"))):
            if not clean:
                raise DatasetError(f"{dest}/images or labels already contain files. Re-run with --clean to replace them.")
        for d in (out_images, out_labels):
            if d.exists():
                shutil.rmtree(d)
        for split in SPLITS:
            (out_images / split).mkdir(parents=True)
            (out_labels / split).mkdir(parents=True)

        for img, lines in usable:
            split = assignment[img.stem]
            shutil.copy2(img, out_images / split / img.name)
            (out_labels / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            summary.imported[split] += 1
            summary.boxes[split] += len(lines)

    if yaml_path is not None:
        write_data_yaml(Path(yaml_path))
    return summary


# --------------------------------------------------------------------------
# Validation / statistics
# --------------------------------------------------------------------------
@dataclass
class SplitReport:
    images: int = 0
    background: int = 0       # images with an empty label file
    boxes: int = 0
    tiny_boxes: int = 0       # box smaller than 1% of image width or height
    issues: list = field(default_factory=list)


def validate_dataset(dest: Path = DATASET_DIR) -> dict[str, SplitReport]:
    dest = Path(dest)
    reports: dict[str, SplitReport] = {}
    for split in SPLITS:
        rep = SplitReport()
        img_dir, lab_dir = dest / "images" / split, dest / "labels" / split
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) if img_dir.exists() else []
        label_stems = {p.stem for p in lab_dir.glob("*.txt")} if lab_dir.exists() else set()
        rep.images = len(images)
        for img in images:
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                rep.issues.append(f"{img.name}: no label file")
                continue
            lines, warns = clean_label_text(lab.read_text(encoding="utf-8"))
            rep.issues += [f"{lab.name}: {w}" for w in warns]
            if not lines:
                rep.background += 1
            rep.boxes += len(lines)
            for ln in lines:
                _, _, _, w, h = (float(v) for v in ln.split())
                if w < 0.01 or h < 0.01:
                    rep.tiny_boxes += 1
        for stem in sorted(label_stems - {p.stem for p in images}):
            rep.issues.append(f"{stem}.txt: label without image")
        reports[split] = rep
    return reports


def print_report(reports: dict[str, SplitReport]) -> None:
    total_images = sum(r.images for r in reports.values())
    print(f"{'split':<7}{'images':>8}{'boxes':>8}{'background':>12}{'tiny boxes':>12}")
    for split, r in reports.items():
        print(f"{split:<7}{r.images:>8}{r.boxes:>8}{r.background:>12}{r.tiny_boxes:>12}")
    print(f"total images: {total_images}")
    if total_images < 500:
        print("NOTE: the project target is ~500-1000 varied labelled images; fewer images means weaker detection.")
    for split, r in reports.items():
        for issue in r.issues[:20]:
            print(f"  [{split}] {issue}")
        if len(r.issues) > 20:
            print(f"  [{split}] ... and {len(r.issues) - 20} more issues")


# --------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset tools for Roblox Entity Tracker.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_imp = sub.add_parser("import-cvat", help="import a CVAT YOLO export (zip or folder)")
    p_imp.add_argument("export", help="path to the CVAT export .zip or extracted folder")
    p_imp.add_argument("--val", type=float, default=0.2, help="validation fraction (default 0.2)")
    p_imp.add_argument("--test", type=float, default=0.1, help="test fraction (default 0.1)")
    p_imp.add_argument("--seed", type=int, default=42)
    p_imp.add_argument("--block", type=int, default=25, help="consecutive images kept together in one split")
    p_imp.add_argument("--clean", action="store_true", help="replace an existing dataset/images and labels")
    p_imp.add_argument("--unlabeled-as-background", action="store_true",
                       help="treat images with no label file as 'no characters here' examples")

    sub.add_parser("stats", help="validate the dataset and print counts")
    sub.add_parser("make-yaml", help="(re)write data.yaml")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "import-cvat":
            s = import_cvat(Path(args.export), val=args.val, test=args.test, seed=args.seed,
                            block=args.block, clean=args.clean,
                            unlabeled_as_background=args.unlabeled_as_background)
            print(f"Images in export: {s.images_found}")
            print("Imported: " + ", ".join(f"{k}={v} ({s.boxes[k]} boxes)" for k, v in s.imported.items()))
            if s.unlabeled_skipped:
                print(f"Skipped {s.unlabeled_skipped} images without a label file "
                      f"(use --unlabeled-as-background if they truly contain no characters).")
            if s.orphan_labels:
                print(f"Ignored {s.orphan_labels} label files with no matching image.")
            for w in s.warnings[:20]:
                print("  warning:", w)
            if len(s.warnings) > 20:
                print(f"  ... and {len(s.warnings) - 20} more warnings")
            print("Wrote data.yaml. Next: python dataset_tools.py stats")
        elif args.cmd == "stats":
            print_report(validate_dataset())
        elif args.cmd == "make-yaml":
            write_data_yaml()
            print(f"Wrote {DATA_YAML}")
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
