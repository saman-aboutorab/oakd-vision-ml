# Progress Log

Tracks bugs, errors, and resolutions encountered during development.

---

## Resolved

### BUG-001 — depthai v3: `XLinkOut` node removed
**File:** `scripts/capture_dataset.py`, `oakd_vision/capture/oakd_capture.py`  
**Error:** `AttributeError: module 'depthai.node' has no attribute 'XLinkOut'`  
**Root cause:** depthai v3 removed `XLinkOut`/`XLinkIn` nodes entirely. Output queues are now created directly on node output objects.  
**Fix:** Replace `xout = pipeline.create(dai.node.XLinkOut); node.out.link(xout.input)` with `q = node.out.createOutputQueue(maxSize=4, blocking=False)`.

---

### BUG-002 — depthai v3: `PresetMode.HIGH_DENSITY` renamed
**File:** `scripts/capture_dataset.py`, `oakd_vision/capture/oakd_capture.py`  
**Error:** `AttributeError: type object 'depthai.node.PresetMode' has no attribute 'HIGH_DENSITY'`  
**Root cause:** depthai v3 renamed StereoDepth preset modes. `HIGH_DENSITY` → `DENSITY`, `HIGH_ACCURACY` → `ACCURACY`.  
**Fix:** Use `dai.node.StereoDepth.PresetMode.DENSITY` and `ACCURACY`.  
**Full v3 preset list:** `ACCURACY, DEFAULT, DENSITY, FACE, FAST_ACCURACY, FAST_DENSITY, HIGH_DETAIL, ROBOTICS`

---

### BUG-003 — depthai v3: `dai.Pipeline(dai.Device())` fails
**File:** `scripts/capture_dataset.py`, `oakd_vision/capture/oakd_capture.py`  
**Error:** `RuntimeError: Couldn't open stream`  
**Root cause:** In v3, `dai.Pipeline()` opens the device internally. Passing an explicit `dai.Device()` as argument causes a double-open conflict.  
**Fix:** Use `dai.Pipeline()` with no arguments. The pipeline connects to the first available device automatically.

---

### BUG-004 — depthai v3: missing `setBoardSocket` on depth alignment
**File:** `scripts/capture_dataset.py` (original user-written version)  
**Symptom:** Depth pixels don't correspond to RGB pixels — wrong depth values when sampling bounding boxes.  
**Root cause:** `stereo.setDepthAlign()` was missing, plus `mono_left/right.setBoardSocket()` was missing (defaults vary by version).  
**Fix:** Added `stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)`, `mono_left.setBoardSocket(CAM_B)`, `mono_right.setBoardSocket(CAM_C)`.

---

### BUG-005 — OAK-D VPU crash with 640×640 stereo depth at 25fps
**File:** `scripts/capture_dataset.py`  
**Error:** `Device with id ... has crashed. Crash dump logs stored in .cache/depthai/crashdumps/`  
**Root cause:** Running stereo depth at 640×640 resolution with DENSITY preset at 25fps exceeded the Myriad X VPU's memory/compute budget, causing the device itself to crash (not just a USB bandwidth issue).  
**Fix:** Removed stereo depth from `capture_dataset.py` entirely. Depth is not needed for YOLO training — the model trains on JPGs only. Depth is only used at live inference time where the pipeline is purpose-built for it.

---

### BUG-005b — USB 2.0 bandwidth crash with RGB + stereo depth
**File:** `scripts/capture_dataset.py`  
**Error:** `Couldn't read data from stream: '__x_3_depth' (X_LINK_ERROR)` — device crashes repeatedly  
**Root cause:** Camera connected via USB-C → USB-A cable to a USB 2.0 port. USB 2.0 (~40 MB/s actual) cannot carry both RGB preview and stereo depth simultaneously (combined ~60 MB/s).  
**Fix:** Script now auto-detects USB speed and falls back to RGB-only mode if USB 2.0 is detected. RGB-only is sufficient for YOLO training (model trains on JPGs only; depth is only needed at inference time).  
**Permanent fix:** Connect to a USB 3.0 port (blue colour, or SS/SuperSpeed label on laptop). The cable type (USB-C → USB-A) is fine — the port on the laptop must be USB 3.0.  
**RPi4 note:** RPi4 has two USB 3.0 (blue) ports — always use those for OAK-D on the robot.

---

### BUG-008 — Camera image orientation: rotation direction confusion
**File:** `scripts/capture_dataset.py`  
**Symptom:** Image appeared sideways; switching between `ROTATE_90_CLOCKWISE` and `ROTATE_90_COUNTERCLOCKWISE` each time caused a 180° flip rather than the expected 90° correction.  
**Root cause:** CW and CCW are 180° apart from each other, so toggling between them flips the image rather than incrementally adjusting. The correct approach is to test from a fixed baseline (no rotation) and add exactly one rotation step.  
**Fix:** Removed all rotation. The OAK-D Lite outputs the correct orientation when mounted in the robot's intended position — no software rotation needed.

---

### BUG-009 — `live_demo.py`: `ModuleNotFoundError: No module named 'oakd_vision'`
**File:** `scripts/live_demo.py`  
**Error:** `ModuleNotFoundError: No module named 'oakd_vision'`  
**Root cause:** The `oakd_vision` package was never installed into the venv. Running scripts directly without installing the package means Python can't find it.  
**Fix:** `pip install -e .` from the project root. The `-e` flag installs in editable mode — changes to the source are reflected immediately without reinstalling.

---

### BUG-010 — `live_demo.py`: window opens tiny and black
**File:** `scripts/live_demo.py`  
**Symptom:** Camera window opened as a small black rectangle, not 640×640.  
**Root cause:** `cv2.imshow()` without `cv2.namedWindow(..., cv2.WINDOW_NORMAL)` lets OpenCV choose the window size, which defaults to a small size before any frame arrives. A placeholder frame was shown but the window size wasn't explicitly set.  
**Fix:** Added `cv2.namedWindow(win, cv2.WINDOW_NORMAL)` + `cv2.resizeWindow(win, 640, 640)` before the first `imshow` call.

---

### BUG-011 — `live_demo.py`: display blinks black every frame
**File:** `scripts/live_demo.py`  
**Symptom:** Live feed flickered between the real image and a black placeholder every ~1 second.  
**Root cause:** When `rgb_q.tryGet()` returns `None` (no new frame available), the code was showing the black placeholder image. At 25fps, frames arrive faster than the display loop so `tryGet()` returns `None` frequently.  
**Fix:** Added `last_display` variable that caches the most recent valid frame. When no new frame arrives, `last_display` is shown instead of the placeholder — smooth, flicker-free display.

---

### BUG-012 — `live_demo.py`: uses depthai v2 API (XLinkOut, ColorCamera, getOutputQueue)
**File:** `scripts/live_demo.py`  
**Error:** Would have crashed at runtime with `AttributeError` on v2 node names.  
**Root cause:** The original `live_demo.py` was written using the v2 API (`ColorCamera`, `MonoCamera`, `XLinkOut`, `dai.Device(pipeline)`) before we discovered depthai v3's breaking changes during P1 capture work.  
**Fix:** Fully rewrote `live_demo.py` using the v3 API — `Camera.build(socket)`, `requestOutput()`, `createOutputQueue()`, `dai.Pipeline(dai.Device())` pattern, consistent with `capture_dataset.py`.

---

### BUG-006 — PKSampler yielding individual indices instead of batches
**File:** `oakd_vision/tracker/triplet_dataset.py`  
**Error:** `TypeError: 'int' object is not iterable` in DataLoader worker  
**Root cause:** `batch_sampler` expects each `__iter__` yield to be a **list** of indices (one full batch). Using `yield from indices` unpacked the list and yielded integers one by one.  
**Fix:** Changed `yield from indices` → `yield indices`.

---

### BUG-007 — ReID training: loss=NaN, active=0% from epoch 1
**File:** `oakd_vision/tracker/losses.py`, `training/configs/reid_config.yaml`  
**Symptom:** All epochs showed `train_loss=nan  active=0.00%  val_loss=nan`. Model weights collapsed to NaN.  
**Root cause 1:** `pairwise_distances()` used `sq_dist.clamp(min=0.0)` then `sqrt()`. When two embeddings are identical early in training, squared distance = 0, and `d(sqrt(x))/dx` at x=0 is infinite → NaN gradient propagates through entire model.  
**Root cause 2:** With 14 train identities and `P=8`, only 1 batch per epoch — too little training signal to escape the NaN regime.  
**Fix 1:** Changed `clamp(min=0.0)` → `clamp(min=1e-12)` to avoid zero inside sqrt.  
**Fix 2:** Reduced `P` from 8 → 6, giving ~2 batches per epoch with 14 identities.

---

## P3 Training Results

### Run 1 — concat, freeze_layers=2, 125 frames (baseline)

**Date:** 2026-04-10  
**Command:** `python -m oakd_vision.fusion.train_fusion --strategy concat`  
**Checkpoint:** `runs/fusion/concat/best.pt` (saved at epoch 2)

| Metric | Value |
|--------|-------|
| Epochs | 60 |
| train_acc (final) | 99.85% |
| val_acc (final) | 79.58% |
| best val_loss | 0.6827 (epoch 2) |
| val_acc_free | 0.93 |
| val_acc_caution | 0.46 |
| val_acc_obstacle | 0.65 |
| val_acc_unknown | 0.86 |

**Diagnosis:** Severe overfitting — train 99.9% vs val 79.6%. Best val_loss was at epoch 2, then model memorized training data.  
**Root causes:**
1. Only 100 train frames (4800 patches) — too little data for 11M parameter model
2. `caution` at 46% — only 5.9% of patches, not enough even with 4.6× class weight
3. `freeze_layers=2` leaves too many ResNet parameters free to overfit on small dataset

**Planned fix:**
- Collect ~200 more frames this weekend (focus on caution scenarios)
- Retrain with `freeze_layers=3` to reduce overfitting
- Run all 3 strategies (concat, attention, gated) for ablation comparison

---

## Open

_None currently._
