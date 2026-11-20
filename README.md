# Roblox Entity Tracker

An **external, visual-only** utility for Windows. It looks at screenshots of the Roblox window, finds humanoid characters (players and NPCs alike, one class: `character`) with a YOLO model you train yourself, follows them with ByteTrack, and draws boxes, temporary IDs and a live counter in a transparent PyQt6 overlay.

It does **not** read game memory, inject code, use executors or touch game files. It only sees what is on screen, so it cannot find characters behind walls or outside the camera view. IDs such as `ID 3` are temporary tracking labels, **not** Roblox usernames or permanent identities.

> Check the rules of any game/server before using an overlay in it. Even a purely visual tool can break a game's rules, especially in competitive play.

## Project layout

```
RobloxEntityTracker/
    run.bat             double-click launcher menu
    main.py             run the app (capture -> detect -> track -> overlay)
    capture.py          Roblox window capture (MSS) + self-test
    collector_gui.py    screenshot collector with a window (buttons, preview)
    collect_dataset.py  terminal version of the collector
    dataset_tools.py    CVAT export -> train/val/test dataset, validation, stats
    train.py            transfer-learning training, saves models/best.pt
    detector.py         YOLO wrapper (GPU with CPU fallback)
    test_detection.py   try the model on screenshots / measure on the test split
    tracker.py          ByteTrack wrapper
    pipeline.py         background detection+tracking thread
    overlay.py          PyQt6 overlay, control panel, hotkey
    config.py           settings (saved to config.json)
    hotkeys.py          global key polling (read-only)
    data.yaml           dataset description
    requirements.txt
    models/             best.pt goes here after training
    dataset/            raw_images/, cvat_export/, images/, labels/
    tests/              pytest unit tests
```

## Quick start: `run.bat`

Double-click **`run.bat`**. On first run it creates the virtual environment and installs the requirements, then shows a menu: collect screenshots, import the CVAT export, check the dataset, train, test, and run the overlay. (For a GPU build of PyTorch see Stage 1 below, installed *before* running `run.bat`.)

## Stage 1 - Environment

Python 3.10-3.12 (64-bit) on Windows 10/11. In PowerShell, inside the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**NVIDIA GPU (recommended for training, optional for running):** `pip install ultralytics` installs a CPU build of PyTorch on Windows. For GPU, install the CUDA build first, using the exact command from https://pytorch.org/get-started/locally/ (for example `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`), then `pip install -r requirements.txt`. Check it with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU everything still works on the CPU; training is just much slower (use a small model such as `yolo11n.pt`).

Run the unit tests any time: `python -m pytest tests -q`

## Stage 2 - Check screen capture

1. Start Roblox in **windowed** or **borderless** mode (exclusive fullscreen often captures as black).
2. Run:

```powershell
python capture.py --seconds 5
```

It prints the window region, size and achieved FPS and saves `capture_test.png`. Open that image: it must show the game. If it says the window was not found, run `python capture.py --list-windows` to see window titles (the Roblox window is normally titled exactly `Roblox`; change `window_title` in `config.json` if yours differs). To capture a whole monitor instead: `python capture.py --monitor`.

## Stage 3 - Build the dataset

### 3a. Collect screenshots

**Easy way (window with buttons):**

```powershell
python collector_gui.py
```

Pick a session name, press **Take screenshot** (or **F7**) or **Start auto capture** (or **F6**). A live preview, saved counter and an **Open folder** button are included. Duplicate-looking frames are skipped in auto mode.

**Terminal version** (same features, no window):

```powershell
python collect_dataset.py --session city1
```

Hotkeys (work while Roblox has focus): **F6** start/stop auto-capture (1 image/second), **F7** save one image now, **END** quit. Near-duplicate frames are skipped automatically. Images land in `dataset/raw_images/` as `city1_00001.jpg`, ...

Aim for **500-1000 varied images**: many different games/maps, avatar styles and body shapes, animations (running, jumping, sitting), near and far characters, bright and dark scenes, crowds and empty scenes, players *and* NPCs. Use a new `--session` name for each play session.

### 3b. Annotate in CVAT

1. Open https://app.cvat.ai (or your own CVAT server) -> **Projects -> Create new project**. Add one label named exactly **`character`** (type *Rectangle*).
2. Create a **Task** in that project and upload the images from `dataset/raw_images/`. (Tasks of ~200-300 images are comfortable.)
3. Open the job and draw a tight rectangle around **every** visible humanoid character, including partly hidden ones and tiny distant ones that you can clearly recognise. Be consistent: if you label a distant character in one image, label it in all. Frames with no characters can simply have no boxes.
   Shortcuts: `N` = new box, `F` = next frame, `Ctrl+S` = save.
4. When finished: **Task menu -> Export task dataset**, format **YOLO 1.1** (or *Ultralytics YOLO Detection*), tick **Save images**, and download the zip into `dataset/cvat_export/`.

### 3c. Import and check

```powershell
python dataset_tools.py import-cvat dataset\cvat_export\task_export.zip
python dataset_tools.py stats
```

This validates labels (single class `0`, boxes clipped to the image), splits **70 % train / 20 % val / 10 % test** (adjust with `--val` / `--test`), and writes `data.yaml`. The split keeps blocks of consecutive screenshots together (`--block 25`) so neighbouring, nearly identical frames never leak between train and validation. Use `--clean` to redo an import. Images without a label file are skipped (use `--unlabeled-as-background` if they truly contain no characters).

## Stage 4 - Train

```powershell
python train.py --check-only      # verify the dataset first
python train.py                   # yolo11n.pt start weights, 100 epochs
```

Useful options: `--epochs 150`, `--model yolo11s.pt` (slower, more accurate), `--imgsz 640`, `--batch 16`, `--device cpu`, `--workers 0` (if Windows dataloader errors occur).

`train.py` refuses to run without a labelled dataset (minimum 50 training images; 500+ recommended). Start weights are downloaded automatically (transfer learning). When it finishes you get:

* `models/best.pt` - the best checkpoint from **your** data
* `models/training_info.json` - precision / recall / mAP measured on your validation and test images
* `runs/<name>/` - Ultralytics curves, confusion matrix and prediction samples

## Stage 5 - Test detection

```powershell
python test_detection.py --images dataset\images\test --out detection_test
python test_detection.py --eval test
```

Open the annotated images in `detection_test/`. Missed characters or false boxes tell you what to collect and label next; add images, re-import with `--clean`, retrain.

## Stage 6-8 - Track and run the overlay

```powershell
python main.py
```

A small **Entity Tracker Controls** window opens; the overlay appears over the Roblox window and follows it when moved or resized.

| Control | What it does |
|---|---|
| Show overlay / **F8** | show or hide the overlay (works while Roblox has focus; change `toggle_hotkey` in config) |
| Overlay opacity | transparency of boxes and labels |
| Confidence threshold | detections below this do not start or drive tracks |
| Capture resolution | downscale frames before detection (lower = faster) |
| Detection rate | detection frames per second |
| Forget lost tracks after | how long a vanished character may come back with the same ID |

Command-line overrides: `--model`, `--device cpu`, `--conf 0.5`, `--detect-fps 10`, `--scale 0.5`, `--monitor`. Settings are saved to `config.json` on exit or with **Save settings**.

### Performance tips

* Lower **Capture resolution** (0.5 is the default) and **Detection rate** first.
* A GPU build of PyTorch gives the biggest speed-up; the app falls back to the CPU on its own if the GPU path fails.
* Capture, detection+tracking and the UI run on separate threads, so a slow model lowers the detection FPS but never freezes the overlay.
* Screen capture is excluded from the overlay itself via `SetWindowDisplayAffinity` (Windows 10 2004+), so the app does not detect its own boxes.

## Troubleshooting

* **"Model not found"** - you have not trained yet; run Stage 4.
* **Black captures** - switch Roblox from exclusive fullscreen to windowed/borderless.
* **Overlay not visible over the game** - use borderless/windowed mode; check the toggle hotkey.
* **Boxes offset** - make sure only one display scaling setting applies; the app uses per-monitor DPI awareness. Report your scaling % if this happens.
* **`F8` does nothing** - another program may be grabbing it; change `toggle_hotkey` in `config.json`.
* **Training is slow** - use `--device 0` with a CUDA PyTorch build, or reduce `--imgsz` / `--epochs`.
