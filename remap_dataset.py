"""Remap a Roboflow YOLO export to the class set this project trains on.

    python remap_dataset.py "<export folder>" dataset\\roboflow_remapped

Default mapping (source class name -> project class):
    enemy -> body,  head -> head,  teammate -> teammate,  objects -> dropped

Every image is kept: an image with no mapped box simply becomes a background
example. The project class order (ids 0, 1, 2 ...) is the order of the targets in
--map, and it must match CLASS_NAMES in config.py - the script prints the line.

Override the mapping, for example to also keep 'objects':
    --map enemy=body head=head teammate=teammate objects=objects

The original export is never modified. Afterwards run:
    python dataset_tools.py import-cvat dataset\\roboflow_remapped --clean
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "valid", "val", "test")
DEFAULT_MAP = ["enemy=body", "head=head", "teammate=teammate"]


def load_names(src: Path) -> list[str]:
    data = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def parse_map(items: list[str]) -> tuple[dict[str, str], list[str]]:
    """['enemy=body', ...] -> ({'enemy': 'body', ...}, ['body', 'head', ...] in first-seen order)."""
    mapping: dict[str, str] = {}
    targets: list[str] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"bad --map entry '{item}', expected source=target")
        source, target = (s.strip() for s in item.split("=", 1))
        mapping[source.lower()] = target
        if target not in targets:
            targets.append(target)
    return mapping, targets


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Remap a YOLO export to this project's classes.")
    p.add_argument("source", help="extracted Roboflow export folder (contains data.yaml, train/, valid/, test/)")
    p.add_argument("dest", help="output folder (must be empty or not exist)")
    p.add_argument("--map", nargs="+", default=DEFAULT_MAP, metavar="SRC=TARGET",
                   help="source class -> project class; unlisted source classes are dropped")
    args = p.parse_args(argv)

    src, dest = Path(args.source), Path(args.dest)
    if not (src / "data.yaml").exists():
        print(f"ERROR: {src / 'data.yaml'} not found. Point to the extracted export folder.")
        return 1
    try:
        mapping, targets = parse_map(args.map)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    names = load_names(src)
    missing = [s for s in mapping if s not in {n.lower() for n in names}]
    if missing:
        print(f"ERROR: {missing} not found in the dataset's classes {names}.")
        return 1
    id_map = {i: targets.index(mapping[n.lower()]) for i, n in enumerate(names) if n.lower() in mapping}

    print("Class mapping:")
    for i, n in enumerate(names):
        print(f"  {i} {n}: " + (f"-> {id_map[i]} ({targets[id_map[i]]})" if i in id_map else "dropped"))

    if dest.exists() and any(dest.iterdir()):
        print(f"ERROR: {dest} is not empty. Choose a new folder or delete it first.")
        return 1
    out_img, out_lab = dest / "images", dest / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    written = background = 0
    boxes: Counter = Counter()
    for split in SPLIT_DIRS:
        img_dir, lab_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lines: list[str] = []
            lab = lab_dir / f"{img.stem}.txt"
            if lab.exists():
                for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                    except ValueError:
                        continue
                    if cls in id_map:
                        lines.append(f"{id_map[cls]} " + " ".join(parts[1:5]))
                        boxes[targets[id_map[cls]]] += 1
            new_name = f"{split}_{img.name}"  # prefix avoids name clashes between splits
            shutil.copy2(img, out_img / new_name)
            (out_lab / f"{Path(new_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written += 1
            if not lines:
                background += 1

    print(f"\nWrote {written} images to {dest}  (none skipped)")
    print(f"  boxes per class: {dict(boxes)}")
    print(f"  background images (no mapped box): {background}")
    print("\nPut this line in config.py (replacing the old CLASS_NAMES):")
    print(f"  CLASS_NAMES = {targets}")
    print("Then: python dataset_tools.py import-cvat " + str(dest) + " --clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
