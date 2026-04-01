# P3 — RGB-D Traversability CNN (coming after P2)
# See README.md → P3 for the full plan.
#
# Module: oakd_vision/fusion/
# What it will contain:
#   rgb_branch.py         — ResNet18 backbone (pretrained, fine-tuned)
#   depth_branch.py       — lightweight custom CNN trained from scratch
#   fusion_model.py       — TraversabilityNet with 3 fusion strategies
#   traversability_dataset.py — patch dataset loader
#   train_fusion.py       — ablation training loop + W&B
#   evaluate_fusion.py    — per-class F1, confusion matrix, ablation table
#   inference.py          — TraversabilityPredictor (frame → costmap grid)
#   data_collector.py     — saves RGB+depth frame pairs from OAK-D
#   annotator.py          — OpenCV grid annotation tool
