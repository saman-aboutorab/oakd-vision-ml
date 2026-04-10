# oakd-vision-ml

Pure Python/PyTorch computer vision library for OAK-D Lite. Zero ROS2 dependency — installable as a pip package and imported by the [turtlebot3-autonomy-stack](https://github.com/saman-aboutorab/turtlebot3-autonomy-stack) ROS2 monorepo and standalone portfolio projects.

```
pip install -e .
```

---

## What This Repo Is

This is the ML engineering core of a robotics perception system built around the [Luxonis OAK-D Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html) depth camera. All PyTorch training code, W&B experiment tracking, trained model weights, and DepthAI inference pipelines live here.

The repo is structured as a single installable package (`oakd_vision`) with independent submodules per project. Each submodule can be used standalone or composed into a full perception pipeline.

**Architecture:**

```
oakd_vision/
├── detector/       # P1 — YOLO fine-tuning + 3D object detection
├── tracker/        # P2 — ReID embedding model + multi-object tracking
├── capture/        # OAK-D DepthAI capture utilities
└── utils/          # Camera intrinsics, depth processing
```

---

## Projects

Projects follow the build plan across three phases. P1 and P2 are Phase A (laptop only). P5, P6, and P7 are Phase B/C robot missions that live in Repo 1 but are fully unblocked by completing P1 and P2 here.

### P1 — YOLO Fine-Tune + 3D Object Detection

**Status:** Phase A (laptop + OAK-D Lite, no robot required)  
**Module:** `oakd_vision/detector/`

Fine-tune YOLOv8n on a custom household object dataset captured with the OAK-D Lite at robot height (~20cm). Export to OpenVINO blob for on-device inference on the Myriad X VPU at ~25 FPS. Fuse 2D detections with stereo depth to produce 3D positions in camera frame.

🎯 What P1 Solves and Why It Matters
The core problem is this: off-the-shelf YOLOv8 is trained on COCO (80 classes, shot from human-height cameras, in well-lit environments). Your TurtleBot3's OAK-D Lite sits about 20cm off the ground, looking UP at furniture and objects from a completely different perspective. COCO-pretrained YOLO will struggle with this — it's never seen a chair leg from ground level, or a shoe from 15cm away, or a coffee mug viewed from below.
Fine-tuning solves the domain gap between COCO's internet photos and your robot's actual visual experience. It also lets you add classes that COCO doesn't have (your specific items, your room's furniture) and optimize the model size for edge deployment on the OAK-D Lite's Myriad X VPU.

### P1 Classes (v1 — 6 classes, trained and deployed)

| Class | Why | COCO overlap? | mAP@50 |
|---|---|---|---|
| `shoe` | Floor obstacle, very common | No — not in COCO | 0.976 |
| `cable` | Floor hazard (charging cables, etc.) | No | 0.777 |
| `chair_leg` | Most dangerous low obstacle | Partial (chair exists, not legs) | 0.785 |
| `mug` | Search target for P6 | Partial as "cup" | 0.995 |
| `remote` | Search target for P6 | Yes — but wrong angle | 0.995 |
| `person_feet` | For person-following P7 | No — COCO has full body only | 0.838 |

**Overall mAP@50: 0.894** (target was ≥ 0.70) — trained locally on RTX 4070 Laptop GPU in 7 minutes.

Classes not in COCO (`shoe`, `cable`, `chair_leg`, `person_feet`) are the highest priority — the pretrained model has never seen them from robot height.

### Capture Instructions

**Requirements:** OAK-D Lite connected to a USB 3.0 port (blue, labelled SS).

```bash
cd ~/projects/Robotics/oakd-vision-ml
source venv/bin/activate
python scripts/capture_dataset.py <class_name>
```

Run once per class. The script resumes numbering if you stop and restart.

**In the camera window:**
- Click the window to focus it
- `SPACE` — save current frame (green flash = confirmed)
- `Q` — quit

**Capture strategy — aim for ~70 shots per class:**

| Variation | What to do |
|---|---|
| Distance | ~20 shots near (0.3–0.7m), ~20 mid (0.7–1.5m), ~20 far (1.5–2.5m) |
| Angle | Front, side, 45°, slightly above, slightly below |
| Background | Floor, table, carpet, wall, multiple surfaces |
| Lighting | Normal, near window (bright), dim lamp, shadow |

Saved to `dataset/raw/<class_name>/` as `.jpg` files (depth not needed for YOLO training).

**Key deliverables:**
- ✅ Labeled dataset: 854 images, 6 classes, captured at robot height with Roboflow labeling
- ✅ Trained YOLOv8n: `runs/detect/runs/train/p1_v1/weights/best.pt` — mAP@50 = 0.894
- ✅ Live demo: OAK-D → detect → 3D localize in real time (`scripts/live_demo.py`)
- ⏳ Export to ONNX → OpenVINO → blob — deferred to P5 (needed only when deploying on robot)

**Files:**

| File | Purpose |
|------|---------|
| `capture/oakd_capture.py` | DepthAI pipeline: RGB + aligned stereo depth capture, saves `.jpg` + `.npy` pairs |
| `capture/dataset_builder.py` | Takes CVAT/Roboflow export → YOLO format with 70/20/10 split |
| `detector/yolo_trainer.py` | Training pipeline with Ultralytics + W&B auto-logging |
| `detector/evaluate.py` | mAP, per-class AP, confusion matrix, PR curves → `evaluation_report.md` |
| `detector/export.py` | PyTorch → ONNX → OpenVINO IR → DepthAI blob + benchmark table |
| `detector/yolo_inference.py` | `YOLODetector` class: unified `detect(frame)` API for VPU and ONNX modes |
| `detector/depth_fusion.py` | 2D detection + stereo depth → `Detection3D` with 3D position |
| `utils/camera.py` | OAK-D Lite intrinsics, `pixel_to_3d(u, v, depth)` |
| `utils/depth.py` | `get_depth_for_bbox()`: median of valid pixels in bbox region |
| `training/configs/yolov8n_custom.yaml` | Dataset config and augmentation strategy |
| `training/notebooks/train_yolo.ipynb` | Colab Pro training notebook |

**Steps:**

1. Create `oakd-vision-ml` repo with `setup.py` and package structure
2. Write OAK-D capture script — RGB + aligned stereo depth, saves `.jpg` + `.npy` pairs
3. Capture 500–800 images of 8–10 household objects at robot height (~20cm) and normal height; vary lighting, angles, distances, backgrounds
4. Label dataset in CVAT or Roboflow (YOLO format)
address: 
curl -L "https://app.roboflow.com/ds/1HfPCybwFq?key=rLFtJVu81z" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
5. Run `dataset_builder.py` to create 70/20/10 train/val/test split
6. Configure `yolov8n_custom.yaml` with dataset path, augmentation strategy, and hyperparameters
7. Train YOLOv8n (~200 epochs on Colab Pro A100); log all runs to W&B
8. Evaluate on test set: mAP@50, mAP@50:95, per-class AP, confusion matrix, PR curves
9. Export best model: PyTorch → ONNX → OpenVINO IR → DepthAI blob (6 SHAVE cores)
10. Benchmark: FPS on Myriad X VPU vs ONNX Runtime on CPU; print comparison table
11. Add depth fusion: bbox center + stereo depth → 3D position in camera frame
12. Run live demo: OAK-D → detect → overlay 3D position (e.g. `"mug: 1.2m at (0.3, -0.1, 1.2)")`

---

### P2 — Custom ReID Embedding Model + Multi-Object Tracking

**Status:** Phase A (laptop + OAK-D Lite, no robot required)  
**Module:** `oakd_vision/tracker/`

Train a custom ResNet18-based ReID embedding network using triplet loss with batch-hard mining. Integrate with a DeepSORT-style tracker (Kalman filter + Hungarian algorithm). Wire into the full pipeline: OAK-D → P1 detect → crop → P2 embed → match → persistent IDs.

This is the **hero project for ML interviews** — metric learning, custom training loop, Kalman filtering, combinatorial optimization, and live tracking demo, all in one coherent pipeline.

---

### Why P2 needs its own data (not Market-1501)

Public ReID datasets (Market-1501, DukeMTMC) are people from overhead surveillance cameras. Your robot sees household objects from 20cm floor height — completely different domain. We collect our own crops using the P1 detector as an auto-cropper, which takes ~1 hour total.

---

### How much data to collect

We use the same 6 classes from P1. For each class we need **multiple distinct instances** — e.g., 3 different shoes, not 30 photos of one shoe.

| Class | Instances to collect | Crops per instance | Total crops |
|---|---|---|---|
| `shoe` | 4 distinct shoes | ~35 each | ~140 |
| `cable` | 4 distinct cables | ~35 each | ~140 |
| `chair_leg` | 4 chairs/tables | ~35 each | ~140 |
| `mug` | 4 distinct mugs | ~35 each | ~140 |
| `remote` | 4 distinct remotes | ~35 each | ~140 |
| `person_feet` | 4 people / outfits | ~35 each | ~140 |
| **Total** | **24 identities** | | **~840 crops** |

For each instance vary: distance, angle, lighting — the model needs to learn that the same object looks different from different viewpoints.

Folder structure after collection:
```
dataset/reid/
├── shoe_001/   (35 crops of shoe instance 1)
├── shoe_002/   (35 crops of shoe instance 2)
├── cable_001/
├── person_001/ (35 crops of person 1's feet/shoes)
...
```

---

### Key deliverables

- ✅ ReID dataset: ~840 crops, 24 identities across 6 classes
- ✅ Trained `ReIDNet`: `runs/reid/best.pt` — Rank-1 ≥ 70%, mAP ≥ 0.50
- ✅ t-SNE plot: clearly separated identity clusters
- ✅ Live demo: persistent colored IDs survive occlusion, FPS > 15

---

### Files

| File | Purpose |
|------|---------|
| `scripts/collect_reid.py` | Runs P1 on live OAK-D → auto-saves crops per detected class |
| `oakd_vision/tracker/reid_model.py` | `ReIDNet`: ResNet18 + FC head + L2 norm → 128-dim embedding |
| `oakd_vision/tracker/triplet_dataset.py` | `TripletDataset` + `PKSampler` for batch-hard mining |
| `oakd_vision/tracker/losses.py` | `BatchHardTripletLoss` implemented from scratch |
| `oakd_vision/tracker/train_reid.py` | Manual training loop, checkpointing, W&B logging |
| `oakd_vision/tracker/evaluate_reid.py` | CMC curve, mAP, t-SNE, top-5 retrieval grid |
| `oakd_vision/tracker/mot_tracker.py` | `MOTTracker`: Kalman + Hungarian + track lifecycle |
| `training/configs/reid_config.yaml` | Hyperparameters: embedding_dim, margin, lr, P, K |
| `scripts/live_tracking.py` | Live demo: OAK-D → detect → track → persistent ID overlays |

---

### Steps

**Phase 1 — Data Collection**
1. Write `scripts/collect_reid.py` — runs P1 detector live, auto-saves crops with 10% padding into `dataset/reid/<class>_<session>/`
2. Collect ~35 crops per identity across 24 identities (~840 total) — vary angle, distance, lighting per session
3. Manually rename session folders to identity folders: `shoe_session_001` → `shoe_001`, `shoe_002`, etc.

**Phase 2 — Model**
4. Write `reid_model.py` — ResNet18 backbone (ImageNet pretrained) → GAP → FC(512→256) → BN → ReLU → Dropout → FC(256→128) → L2 normalize
5. Write `losses.py` — `BatchHardTripletLoss` from scratch: for each anchor, find hardest positive and hardest negative in the batch
6. Write `triplet_dataset.py` — `PKSampler` samples P=8 identities × K=4 crops per batch (32 samples → 32³ possible triplets, batch-hard picks the worst ones)

**Phase 3 — Training**
7. Write `train_reid.py` — manual loop: forward pass → loss → backward → gradient clip → step; W&B logs loss + LR every epoch
8. Train 50–100 epochs on RTX 4070; checkpoint best Rank-1 model

**Phase 4 — Evaluation**
9. Write `evaluate_reid.py` — CMC curve (Rank-1/5/10), mAP, t-SNE plot of all embeddings, top-5 retrieval grid (query → 5 nearest, green/red border = correct/wrong)
10. Target: Rank-1 ≥ 70%, mAP ≥ 0.50

**Phase 5 — Tracker**
11. Write `mot_tracker.py` — Kalman filter per track (predicts next position), Hungarian algorithm assigns detections to tracks, track lifecycle: tentative → confirmed → deleted
12. Cost matrix = 0.5 × IoU distance + 0.5 × ReID cosine distance

**Phase 6 — Live Demo**
13. Write `scripts/live_tracking.py` — OAK-D → P1 detect → ReID embed → MOTTracker → draw persistent colored boxes + ID labels + trailing path dots
14. Target: FPS > 15 with full pipeline on laptop GPU

---

### P3 — RGB-D Traversability CNN (Architecture Design Showcase)

**Status:** Planned — after P2, before or alongside P5  
**Module:** `oakd_vision/fusion/`

Design a dual-branch CNN that fuses RGB and aligned depth to predict **traversability** for each patch of the camera frame: `free / obstacle / caution / unknown`. The output feeds into Nav2 as a semantic costmap layer, giving the robot context-aware path planning — it knows a carpet is drivable and a staircase is not, even when raw depth geometry is ambiguous.

This is the **architecture design showcase**: you design the network, implement three fusion strategies, run a full ablation study, and justify every decision with data.

**Why dual-branch?**
- **RGB branch** — pretrained ResNet18. Learns texture + semantics (what a staircase *looks like* vs what a carpet looks like)
- **Depth branch** — lightweight custom CNN, trained from scratch on OAK-D depth maps. Learns geometry (surface normals, slopes, gap sizes)
- Neither branch alone is sufficient. RGB can't measure distance; depth can't distinguish carpet from stairs

**Three fusion strategies (ablation study):**

| Strategy | How it works |
|----------|-------------|
| Concatenation | Concatenate 256-dim RGB + 256-dim depth → 512-dim FC head. Baseline. |
| Attention gating | A small network outputs a per-modality weight. In dark rooms it up-weights depth; on reflective surfaces it up-weights RGB. |
| Gated fusion | Like attention but element-wise — each feature dimension independently decides RGB vs depth contribution. |

**Key deliverables:**
- Labeled dataset: 500–800 RGB-D frames, grid-labeled at 8×6 patches per frame (~25K patch samples)
- Ablation table: 5 models (RGB-only, depth-only, concat, attention, gated) with per-class F1 + IoU
- W&B: training curves and comparative charts across all variants
- Custom Nav2 costmap plugin: semantic traversability layer stacked on top of depth/LiDAR
- Standalone spin-off: FastAPI endpoint accepting paired RGB-D images → traversability map

**Files:**

| File | Purpose |
|------|---------|
| `fusion/data_collector.py` | Saves synchronized RGB + depth frames to disk (drives robot manually) |
| `fusion/annotator.py` | Grid-based OpenCV annotation tool — click patches to label free/obstacle/caution/unknown |
| `fusion/traversability_dataset.py` | Loads patch crops from labeled frames, paired RGB + depth |
| `fusion/rgb_branch.py` | ResNet18 backbone (pretrained), fine-tune later layers only |
| `fusion/depth_branch.py` | Lightweight custom CNN, 4–5 conv blocks, trained from scratch |
| `fusion/fusion_model.py` | `TraversabilityNet`: combines both branches with selectable fusion strategy |
| `fusion/train_fusion.py` | Training loop with W&B logging, runs all 5 ablation variants |
| `fusion/evaluate_fusion.py` | Per-class F1, confusion matrix, ablation comparison table |
| `fusion/inference.py` | `TraversabilityPredictor`: runs model on a frame, returns patch grid + costmap values |
| `training/configs/fusion_config.yaml` | Hyperparameters: patch_size, fusion_strategy, lr, backbone_freeze_layers |

**Nav2 integration (in Repo 1):**

The inference node in `turtlebot3-autonomy-stack` calls `oakd_vision.fusion.predict()` and publishes a costmap layer:

```
free      → cost 0    (drive normally)
caution   → cost 100  (planner avoids if possible)
obstacle  → cost 254  (lethal — never cross)
unknown   → cost 128  (moderate — cross only if no better route)
```

**Steps:**

1. ✅ Write `collect_traversability.py`: save synchronized RGB + aligned depth pairs at ~1 FPS using OAK-D handheld (USB 3.0 for depth) — no robot needed for data collection
2. ✅ Write `annotate_traversability.py`: keyboard-driven 8×6 grid label tool; saves labels as JSON alongside each frame
3. ⏳ Collect 500–800 frames — **125 labeled so far** (Run 1 done, Run 2–4 pending this weekend — need more `caution` examples)
4. ✅ Build `TraversabilityDataset`: loads paired RGB crop + depth crop per patch (125 frames → 6000 patches)
5. ✅ Build `rgb_branch.py` (ResNet18 pretrained) and `depth_branch.py` (custom CNN from scratch)
6. ✅ Implement all three fusion strategies in `fusion_model.py`: concat, attention, gated
7. ⏳ Train ablation study — **concat run 1 complete** (see results below); retrain all 3 strategies after weekend data collection with tuned freeze layers
8. Evaluate: per-class F1, confusion matrices; pick winning strategy → `evaluate_fusion.py` (TODO)
9. Export winning model to ONNX
10. Write Nav2 costmap plugin in Repo 1; test semantic layer on robot
11. Record demo: robot correctly treats carpet as free, obstacle as lethal
12. Extract FastAPI spin-off for portfolio

**Ablation results (to be updated after each run):**

| Run | Strategy | freeze_layers | Frames | val_acc | F | C | O | U | Notes |
|-----|----------|--------------|--------|---------|---|---|---|---|-------|
| 1 | concat | 2 | 125 | 79.6% | 0.93 | 0.46 | 0.65 | 0.86 | Overfitting (train=99.9%). Best val_loss=0.68 at epoch 2. More data + higher freeze needed. |
| — | attention | — | — | — | — | — | — | — | Pending |
| — | gated | — | — | — | — | — | — | — | Pending |

**Next training run (this weekend):**
- Collect ~200 more frames (focus on caution: carpet edges, cables, thresholds)
- Label with `--only-unlabeled`
- Retrain all 3 strategies with `freeze_layers=3` (reduce overfitting)
- Compare ablation table above

---

### Data Collection Protocol

**What to photograph and how to label it**

#### Category 1 — Floor Surface Types (`free`)

The most important and most underrepresented class. Without lots of `free` examples the model becomes paranoid and marks everything as obstacle.

| What to photograph | Distance | Label | Notes |
|---|---|---|---|
| Bare hardwood / tile | 0.3m – 3m | `free` | Multiple lighting conditions |
| Carpet / rug (uniform) | 0.3m – 3m | `free` | Different carpet colours/textures |
| Carpet edge / rug boundary | 0.3m – 1m | `caution` | Slight lip can catch wheels |
| Doorway threshold | 0.3m – 1m | `caution` | Small bump |
| Floor transition (tile→carpet) | 0.5m – 1.5m | `caution` | |

#### Category 2 — True Floor Obstacles (`obstacle` / `caution`)

Photograph these **at the distances the robot will actually encounter them** — approaching distances, not from across the room.

| Object | Distance | Label | Notes |
|---|---|---|---|
| Cable lying flat | 0.3m – 1m | `caution` | Beyond 1m cable is invisible to camera |
| Cable bundle / adapter | 0.3m – 0.8m | `obstacle` | Thicker, higher risk |
| Shoe (lying flat) | 0.4m – 1.2m | `obstacle` | |
| Shoe (standing upright) | 0.4m – 1.5m | `obstacle` | Taller, visible further |
| Chair leg (approaching) | 0.3m – 1m | `obstacle` | Beyond 1m depth resolves it better than RGB |
| Mug on floor | 0.3m – 1m | `obstacle` | |
| Backpack on floor | 0.5m – 2m | `obstacle` | Large, visible far |
| Box on floor | 0.5m – 2m | `obstacle` | |

No stairs — not present in this environment. Omit `lethal` class.

#### Category 3 — Distant Objects (`unknown`)

**Key rule:** if an object is beyond 1.5m, do not force an `obstacle` label on the floor patch — label it `unknown`. The robot will approach and re-evaluate.

- Label near floor patches (bottom of frame) as `free` — that floor IS free right now
- Label mid-frame patches where a distant object appears as `unknown` — something is there but floor projection is ambiguous
- Do NOT label distant patches as `obstacle` — the floor the robot is currently on is free, and training with distant-obstacle labels will confuse the model

The system runs inference every frame. Distance ambiguity is handled by `unknown` cost (128), not `obstacle` (254). The model re-evaluates as the robot approaches.

#### Structured Capture Runs

**Run 1 — Pure floor surfaces (1 hour)**
Walk slowly across every floor surface in the home: carpet, tile, wood, transitions. This is the `free` class bulk data. Most important run.

**Run 2 — Obstacle approach sequences (1 hour)**
For each obstacle type (shoe, cable, mug, chair), place it on the floor and walk toward it from 2m away, capturing frames at regular intervals. Captures the full approach sequence: `unknown` → `caution` → `obstacle` as distance closes.

**Run 3 — Cluttered real-world scenes (30 min)**
Natural home scenes with multiple objects — cables and mugs together, real furniture arrangements, objects partially occluded.

**Run 4 — Lighting variations (15 min)**
Repeat key scenes under different lighting conditions (blinds open/closed, overhead light on/off). The RGB branch is sensitive to lighting; without variation it will overfit to your home's typical illumination.

---

## Phase B — Robot Integration (Weeks 9–11, when LiDAR arrives)

### P5 — SLAM + Nav2 + Deploy P1/P2 on Robot

**Repo:** [turtlebot3-autonomy-stack](https://github.com/saman-aboutorab/turtlebot3-autonomy-stack) → `src/tb3_perception_3d/`, `src/tb3_tracking/`  
**Depends on:** P1 (`YOLODetector`, `DepthFusion`) + P2 (`MOTTracker`) from this library

Fix the LDS-01 LiDAR driver, build a SLAM map with SLAM Toolbox, configure Nav2 for autonomous navigation, then mount the OAK-D Lite on the TurtleBot3 and deploy the trained models. Two thin ROS2 nodes wrap this library:

- `tb3_perception_3d`: subscribes to OAK-D RGB + depth → calls `oakd_vision.detector.detect()` → fuses with depth → transforms to map frame via TF2 → publishes `/detections_3d`
- `tb3_tracking`: subscribes to `/detections_3d` → calls `oakd_vision.tracker.update()` → publishes `/tracked_objects`

The ROS2 nodes contain only pub/sub and TF2 glue. All ML inference runs from this library.

```python
# On the robot (RPi4):
pip install -e /path/to/oakd-vision-ml

# Inside the ROS2 wrapper nodes:
from oakd_vision.detector import YOLODetector, DepthFusion
from oakd_vision.tracker import MOTTracker
```

**Steps:**

1. Fix LDS-01 LiDAR driver (diagnose baud rate / USB-serial chip, test `ldlidar_stl_ros2`)
2. Run SLAM Toolbox → drive robot manually with teleop → build and save map of room
3. Configure Nav2: costmap, planner, controller; test autonomous navigation to goal via RViz2
4. Mount OAK-D Lite on TurtleBot3 (facing forward, ~10–15° down tilt, ~20–25cm height); route USB-C to RPi4 USB 3.0 port
5. Update TurtleBot3 URDF: add `oak_d_link` transform from `base_link`, verify TF tree in RViz2
6. `pip install -e oakd-vision-ml` on RPi4; verify camera streams at robot bandwidth
7. Write `tb3_perception_3d` ROS2 node: OAK-D RGB + depth → `detect()` → depth fusion → TF2 transform to map frame → publish `/detections_3d`
8. Write `tb3_tracking` ROS2 node: `/detections_3d` → `tracker.update()` → publish `/tracked_objects`
9. Test full pipeline on robot: maps, detects, tracks objects in real environment

---

## Phase C — Robot Missions (Weeks 12–13)

### P7 — Person Following

**Repo:** [turtlebot3-autonomy-stack](https://github.com/saman-aboutorab/turtlebot3-autonomy-stack) → `src/tb3_missions/person_follow_node.py`  
**Depends on:** P2 (`/tracked_objects` topic from `tb3_tracking`)

State machine: `SEARCHING → LOCKED → FOLLOWING → LOST`. Subscribes to `/tracked_objects`, locks onto a person track by ID, and runs a PID controller to maintain ~1.5m distance using the 3D position from the P2 tracker. Safety: stops if obstacle detected between robot and person; re-acquires if target lost for < 30s.

This is the primary demo video target — 60s person-following video uploaded to YouTube.

### P6 — Object Search

**Repo:** [turtlebot3-autonomy-stack](https://github.com/saman-aboutorab/turtlebot3-autonomy-stack) → `src/tb3_missions/`  
**Depends on:** P1 (`/detections_3d` topic from `tb3_perception_3d`) + Nav2

Exploration behavior: rotate and scan with the P1 detector. When target object class is detected, navigate to it with Nav2 and confirm arrival.

**P7 steps:**

1. Subscribe to `/tracked_objects`; implement state machine: `SEARCHING → LOCKED → FOLLOWING → LOST`
2. Lock onto first detected person track; store `target_track_id`
3. PID controller: linear velocity proportional to `(distance - 1.5m)`, angular velocity proportional to horizontal offset
4. Safety: stop if obstacle between robot and person; stop if distance < 0.5m
5. Re-acquisition: if target lost > 3s enter `LOST`, rotate to search; timeout 30s → `SEARCHING` for new target
6. Tune PID gains, record 60s person-following demo video

**P6 steps:**

1. Exploration behavior: rotate in place, scan with P1 detector for target class
2. When target detected: send Nav2 goal to its 3D position
3. Confirm arrival with class check
4. Record 60s object search demo video; upload both videos to YouTube and link from READMEs

---

## Installation

```bash
git clone https://github.com/saman-aboutorab/oakd-vision-ml
cd oakd-vision-ml
python3 -m venv venv
source venv/bin/activate      # always activate before running any script
pip install -e .
```

> **Every terminal session:** run `source venv/bin/activate` first.  
> Your prompt will show `(venv)` when it's active. Scripts will fail with  
> `ModuleNotFoundError` if you forget.

**Dependencies:** `depthai>=3.0`, `opencv-python>=4.8`, `numpy`, `torch>=2.0`, `ultralytics>=8.0`, `wandb`, `onnxruntime`

**Verify:**
```python
import oakd_vision
from oakd_vision.detector import YOLODetector
from oakd_vision.tracker import MOTTracker
```

---

## Repository Strategy

This repo is **Repo 2** in a 3-repo architecture:

| Repo | Purpose |
|------|---------|
| [turtlebot3-autonomy-stack](https://github.com/saman-aboutorab/turtlebot3-autonomy-stack) | ROS2 monorepo — robot nodes, Nav2, SLAM, missions. Imports this lib. |
| **oakd-vision-ml** (this repo) | Pure Python ML library — all training, inference, and model weights |
| vision-ml-portfolio (planned) | Standalone spin-offs with FastAPI endpoints, Dockerfiles, HuggingFace demos |

The separation means: robotics engineers clone Repo 1, ML/CV engineers clone this repo, and the portfolio is accessible to anyone via live demos.

---

## Experiment Tracking

All training runs are logged to [Weights & Biases](https://wandb.ai).

```bash
export WANDB_PROJECT=oakd-vision-ml
python -m oakd_vision.detector.yolo_trainer train --config training/configs/yolov8n_custom.yaml
python -m oakd_vision.tracker.train_reid --config training/configs/reid_config.yaml
```

---

## Hardware

| Component | Spec |
|-----------|------|
| Camera | Luxonis OAK-D Lite (RGB 12MP + stereo 800P + Myriad X VPU) |
| Training | Colab Pro (A100) or local GPU |
| Robot deployment | Raspberry Pi 4 (4GB), connected via USB-C to USB-A 3.0 |
