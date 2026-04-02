# P1 Progress Log

Tracks bugs, errors, and resolutions encountered during P1 development.

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

## Open

_None currently._
