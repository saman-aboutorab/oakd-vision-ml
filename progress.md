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

## Open

_None currently._
