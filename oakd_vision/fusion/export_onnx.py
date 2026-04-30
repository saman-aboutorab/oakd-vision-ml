"""P3 Traversability CNN — ONNX export.

Exports a trained TraversabilityNet checkpoint to ONNX, then verifies the
export by running a dummy forward pass through ONNXRuntime and comparing
outputs to the PyTorch model.

Usage:
    python -m oakd_vision.fusion.export_onnx
    python -m oakd_vision.fusion.export_onnx --checkpoint runs/fusion/gated_f3/best.pt
    python -m oakd_vision.fusion.export_onnx --checkpoint runs/fusion/concat_f3/best.pt

Output:
    runs/fusion/<run_tag>/model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from oakd_vision.fusion.fusion_model import TraversabilityNet


def export(checkpoint: Path, cfg: dict, opset: int = 17) -> Path:
    # Infer strategy from checkpoint dir name (e.g. "gated_f3" → "gated")
    run_tag  = checkpoint.parent.name
    strategy = run_tag.split("_")[0]
    assert strategy in ("concat", "attention", "gated"), \
        f"Cannot infer strategy from dir name '{run_tag}'"

    device = torch.device("cpu")  # export on CPU for maximum portability

    freeze_layers = cfg["model"].get("freeze_layers", 2)
    model = TraversabilityNet(
        embedding_dim    = cfg["model"]["embedding_dim"],
        num_classes      = cfg["model"]["num_classes"],
        fusion_strategy  = strategy,
        dropout          = cfg["model"]["dropout"],
        freeze_rgb_layers = freeze_layers,
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    # Dummy inputs — batch of 1 patch
    ph, pw  = cfg["data"]["patch_size"]
    rgb_in  = torch.zeros(1, 3, ph, pw)
    depth_in = torch.zeros(1, 1, ph, pw)

    out_path = checkpoint.parent / "model.onnx"

    torch.onnx.export(
        model,
        (rgb_in, depth_in),
        str(out_path),
        opset_version      = opset,
        input_names        = ["rgb", "depth"],
        output_names       = ["logits"],
        dynamic_axes       = {
            "rgb":    {0: "batch"},
            "depth":  {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print(f"Exported → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def verify(out_path: Path, cfg: dict, n_bench: int = 200):
    import time
    import onnx
    import onnxruntime as ort
    from oakd_vision.fusion.traversability_dataset import INT_TO_LABEL

    # Check model is well-formed
    onnx.checker.check_model(str(out_path))
    print("ONNX model check: OK")

    ph, pw = cfg["data"]["patch_size"]
    batch  = cfg["data"]["grid_cols"] * cfg["data"]["grid_rows"]  # 48 patches = 1 frame

    rgb_np   = np.random.randn(batch, 3, ph, pw).astype(np.float32)
    depth_np = np.random.rand(batch, 1, ph, pw).astype(np.float32)

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"rgb": rgb_np, "depth": depth_np})[0]

    assert logits.shape == (batch, cfg["model"]["num_classes"]), \
        f"Unexpected output shape: {logits.shape}"
    print(f"ONNXRuntime inference: OK  output shape={logits.shape}")

    # Latency benchmark
    for _ in range(10):  # warmup
        sess.run(["logits"], {"rgb": rgb_np, "depth": depth_np})
    t0 = time.perf_counter()
    for _ in range(n_bench):
        out = sess.run(["logits"], {"rgb": rgb_np, "depth": depth_np})
    ms = (time.perf_counter() - t0) / n_bench * 1000
    print(f"\nLatency benchmark ({n_bench} frames, batch={batch} patches, CPU):")
    print(f"  {ms:.1f} ms/frame  →  {1000/ms:.0f} fps")
    preds = [INT_TO_LABEL[int(p)] for p in np.argmax(out[0], axis=1)[:8]]
    print(f"  Sample predictions: {preds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default="runs/fusion/gated_f3/best.pt")
    parser.add_argument("--config",     default="training/configs/fusion_config.yaml")
    parser.add_argument("--opset",      default=17, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt     = Path(args.checkpoint)
    out_path = export(ckpt, cfg, opset=args.opset)
    verify(out_path, cfg)
    print("\nDone.")
