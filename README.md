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

**Key deliverables:**
- Labeled dataset: 500–800 images, 8–10 object classes, captured at robot height and normal height
- Trained YOLOv8n: `models/best.pt`, `models/best.onnx`, `models/best.blob`
- W&B dashboard: loss curves, mAP@50, mAP@50:95, per-class AP, confusion matrix
- Live demo: OAK-D → detect → 3D localize in real time
- Benchmark: ~25 FPS on Myriad X VPU vs ~5–8 FPS ONNX on CPU

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

Train a custom ResNet18-based embedding network from scratch using triplet loss with batch-hard mining. Integrate with a DeepSORT-style tracker using Kalman filtering and the Hungarian algorithm. Replace the default ReID extractor in DeepSORT with the trained model. Wire into the full pipeline: OAK-D → P1 detect → crop → P2 embed → Hungarian match → persistent 3D tracks.

This is the hero project for ML interviews — end-to-end metric learning, custom training loop, standard ReID evaluation protocol, and live tracking demo.

**Key deliverables:**
- ReID dataset: 2,000–5,000 crops organized by identity (`reid_dataset/<identity_id>/`)
- Trained model: `models/reid_best.pt`, `models/reid_best.onnx`
- Evaluation: CMC curve (Rank-1 > 70%), mAP > 0.5, t-SNE embedding visualization
- W&B: loss curves, Rank-1/mAP logged every 10 epochs
- Live demo: persistent IDs survive occlusions, FPS > 15 with full pipeline

**Files:**

| File | Purpose |
|------|---------|
| `tracker/data_collector.py` | Runs P1 detector on live OAK-D → crops detections → IoU-based identity assignment |
| `tracker/reid_model.py` | `ReIDNet`: ResNet18 backbone + custom head + L2 normalization |
| `tracker/triplet_dataset.py` | `TripletDataset` + `PKSampler` for P×K batch-hard mining |
| `tracker/losses.py` | `TripletLoss` + `BatchHardTripletLoss` implemented from scratch |
| `tracker/train_reid.py` | Manual training loop (no high-level API), checkpointing, W&B logging |
| `tracker/evaluate_reid.py` | CMC curve, mAP, t-SNE visualization, top-5 retrieval grid |
| `tracker/mot_tracker.py` | `MOTTracker`: Kalman predict → cost matrix → Hungarian assign → track lifecycle |
| `tracker/analytics.py` | `TrackingAnalyzer`: heatmaps, path visualization, dwell time, zone counting |
| `training/configs/reid_config.yaml` | Hyperparameters: embedding_dim, margin, lr, P, K, mining strategy |

**Architecture decisions (documented in module):**
- ResNet18: lightweight enough for real-time, strong pretrained features
- 128-dim embedding: standard in ReID literature, good accuracy/speed tradeoff
- L2 normalization: forces embeddings onto unit hypersphere; cosine distance = L2 distance
- Batch-hard mining: selects hardest positive and hardest negative per anchor in each batch

**Steps:**

1. Write `data_collector.py`: run P1 detector on live OAK-D feed → crop each detection (+ 10% padding) → resize to 128×64 → assign identity via consecutive-frame IoU matching
2. Collect 2,000–5,000 crops across 100+ identities; optionally augment with Market-1501 for pretraining
3. Design `ReIDNet`: ResNet18 backbone (pretrained ImageNet) → Global Average Pooling → FC 512→256 → BN → ReLU → Dropout → FC 256→128 → L2 normalize
4. Implement `TripletLoss` and `BatchHardTripletLoss` from scratch (no library)
5. Implement `TripletDataset` with `PKSampler` (P=8 identities × K=4 images per batch)
6. Write manual training loop (no high-level API): forward all three branches, compute loss, backward, clip gradients, step scheduler; log to W&B every epoch
7. Train 50–100 epochs; log loss curves and learning rate schedule to W&B
8. Evaluate with standard ReID protocol: CMC curve (Rank-1, Rank-5, Rank-10), mAP
9. Visualize embedding space with t-SNE — same-identity clusters should be visible
10. Generate top-5 retrieval grid: query → top-5 nearest matches, green/red borders
11. Compare your model vs an off-the-shelf ReID baseline; document delta in README
12. Export trained model to ONNX
13. Integrate into `MOTTracker`: Kalman predict → appearance + motion cost matrix → Hungarian assignment → track lifecycle (create, update, delete)
14. Add 3D position and velocity estimation per track from stereo depth
15. Run live OAK-D demo: persistent colored bounding boxes per ID + trailing paths + FPS counter
16. Measure tracking metrics: MOTA, IDF1, ID switch count
17. Record demo video of live tracking with overlays
18. Build `TrackingAnalyzer`: heatmap generation, path visualization, dwell time computation

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

1. Write `data_collector.py`: save synchronized RGB + aligned depth pairs at ~1 FPS while driving robot manually
2. Write `annotator.py`: OpenCV grid tool — click 8×6 cells to label; saves labels as JSON alongside each frame
3. Collect 500–800 frames; aim for variety: carpet, bare floor, thresholds, furniture, cables, stairs
4. Build `TraversabilityDataset`: loads paired RGB crop + depth crop per patch
5. Build `rgb_branch.py` (ResNet18, freeze first 2 layers) and `depth_branch.py` (custom CNN)
6. Implement all three fusion strategies in `fusion_model.py` as selectable modes
7. Train ablation study: all 5 variants, log to W&B, 50–100 epochs each
8. Evaluate: per-class F1, confusion matrices; pick winning strategy
9. Export winning model to ONNX
10. Write Nav2 costmap plugin in Repo 1; test semantic layer on robot
11. Record demo: robot correctly treats carpet as free, staircase as lethal
12. Extract FastAPI spin-off for portfolio

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
pip install -e .
```

**Dependencies:** `depthai>=2.24`, `opencv-python>=4.8`, `numpy`, `torch>=2.0`, `ultralytics>=8.0`, `wandb`, `onnxruntime`

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
