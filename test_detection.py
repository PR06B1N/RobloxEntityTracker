"""Stage 5 - try the trained detector on screenshots and/or measure it on the test split.

    python test_detection.py --images dataset/images/test --out detection_test
    python test_detection.py --eval test          # real metrics on the held-out test images

Nothing here is simulated: boxes come from models/best.pt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from config import DATA_YAML, MODELS_DIR, ROOT
from detector import Detections, Detector, ModelNotFoundError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_detections(image: np.ndarray, det: Detections) -> np.ndarray:
    out = image.copy()
    for (x1, y1, x2, y2), c in zip(det.xyxy, det.conf):
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(out, f"{c:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Test the character detector.")
    p.add_argument("--model", default=str(MODELS_DIR / "best.pt"))
    p.add_argument("--images", help="an image file or a folder of images")
    p.add_argument("--out", default="detection_test", help="folder for annotated images")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=50, help="max number of images to process")
    p.add_argument("--eval", choices=["val", "test"], help="run the Ultralytics evaluation on this split")
    args = p.parse_args(argv)

    if not args.images and not args.eval:
        p.error("give --images and/or --eval")

    try:
        detector = Detector(args.model, device=args.device, imgsz=args.imgsz)
    except ModelNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Model loaded on {detector.describe()}")

    if args.images:
        files = collect_images(Path(args.images))[: args.limit]
        if not files:
            print("No images found.")
            return 1
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        times, total = [], 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"  cannot read {f}")
                continue
            t0 = time.perf_counter()
            det = detector.detect(img, args.conf, args.iou)
            times.append(time.perf_counter() - t0)
            total += len(det)
            cv2.imwrite(str(out_dir / f.name), draw_detections(img, det))
            print(f"  {f.name}: {len(det)} character(s)")
        if times:
            print(f"{len(times)} images, {total} detections, {1000 * np.mean(times):.1f} ms/image average. Annotated copies in {out_dir}/")
            print("Open them and check: missed characters? boxes on non-characters? Those are your next labelling targets.")

    if args.eval:
        from train import build_runtime_yaml
        runtime = build_runtime_yaml(DATA_YAML, ROOT / "runs" / "runtime_data.yaml")
        m = detector.model.val(data=str(runtime), split=args.eval, device=detector.device, plots=False, verbose=False)
        print(f"{args.eval}: precision={m.box.mp:.3f} recall={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
