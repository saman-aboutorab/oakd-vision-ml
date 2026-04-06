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

### BUG-005 — USB 2.0 bandwidth crash with RGB + stereo depth
**File:** `scripts/capture_dataset.py`  
**Error:** `Couldn't read data from stream: '__x_3_depth' (X_LINK_ERROR)` — device crashes repeatedly  
**Root cause:** Camera connected via USB-C → USB-A cable to a USB 2.0 port. USB 2.0 (~40 MB/s actual) cannot carry both RGB preview and stereo depth simultaneously (combined ~60 MB/s).  
**Fix:** Script now auto-detects USB speed and falls back to RGB-only mode if USB 2.0 is detected. RGB-only is sufficient for YOLO training (model trains on JPGs only; depth is only needed at inference time).  
**Permanent fix:** Connect to a USB 3.0 port (blue colour, or SS/SuperSpeed label on laptop). The cable type (USB-C → USB-A) is fine — the port on the laptop must be USB 3.0.  
**RPi4 note:** RPi4 has two USB 3.0 (blue) ports — always use those for OAK-D on the robot.

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

## Open

_None currently._
