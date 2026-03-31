"""Export a trained YOLOv8 model to ONNX → OpenVINO IR → DepthAI blob.

Export chain:
    best.pt  →  best.onnx  →  best_openvino/  →  best.blob (Myriad X, 6 SHAVEs)

The blob is the final artifact loaded by YOLODetector(mode="vpu"). The ONNX
file is used for CPU benchmarking and as the OpenVINO IR source.

After exporting, a benchmark table is printed comparing ONNX CPU vs VPU FPS.

Usage:
    python -m oakd_vision.detector.export \\
        --model models/best.pt \\
        --output models/ \\
        --shaves 6

Requirements:
    pip install 'oakd_vision[openvino,blob]'
    (openvino for IR export, blobconverter for blob conversion)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def export(
    model_path: str,
    output_dir: str = "models",
    shaves: int = 6,
    imgsz: int = 640,
    benchmark_frames: int = 100,
    skip_blob: bool = False,
) -> dict:
    """Run the full export chain and benchmark.

    Args:
        model_path: path to best.pt
        output_dir: where to copy final artifacts
        shaves: number of SHAVE cores for the blob (1–6, 6 = max speed)
        imgsz: model input resolution
        benchmark_frames: number of frames for CPU FPS benchmark
        skip_blob: skip blob conversion (requires physical OAK-D device online)

    Returns:
        Dict with paths to exported artifacts.
    """
    from ultralytics import YOLO

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    artifacts = {}

    # ------------------------------------------------------------------ ONNX
    print("\n[1/3] Exporting to ONNX...")
    onnx_export = model.export(format="onnx", imgsz=imgsz, simplify=True)
    onnx_path = Path(onnx_export)
    final_onnx = out / "best.onnx"
    onnx_path.replace(final_onnx)
    artifacts["onnx"] = final_onnx
    print(f"    → {final_onnx}")

    # --------------------------------------------------------------- OpenVINO
    print("\n[2/3] Exporting to OpenVINO IR (FP16)...")
    ov_export = model.export(format="openvino", imgsz=imgsz, half=True)
    ov_dir = Path(ov_export)
    xml_files = list(ov_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml found in OpenVINO output: {ov_dir}")
    ir_xml = xml_files[0]
    ir_bin = ir_xml.with_suffix(".bin")
    artifacts["openvino_xml"] = ir_xml
    artifacts["openvino_bin"] = ir_bin
    print(f"    → {ir_xml}")

    # ------------------------------------------------------------------ Blob
    if not skip_blob:
        print(f"\n[3/3] Converting to DepthAI blob ({shaves} SHAVEs)...")
        try:
            import blobconverter
            blob_path = blobconverter.from_openvino(
                xml=str(ir_xml),
                bin=str(ir_bin),
                data_type="FP16",
                shaves=shaves,
                version="2022.1",
            )
            final_blob = out / "best.blob"
            Path(blob_path).replace(final_blob)
            artifacts["blob"] = final_blob
            print(f"    → {final_blob}")
        except Exception as e:
            print(f"    Blob conversion failed: {e}")
            print("    You can convert manually at: https://blobconverter.luxonis.com")
    else:
        print("\n[3/3] Blob conversion skipped (--skip-blob)")

    # --------------------------------------------------------------- Benchmark
    print("\n[Benchmark] ONNX Runtime on CPU...")
    fps_onnx = _benchmark_onnx(str(final_onnx), imgsz, benchmark_frames)
    print(f"    ONNX CPU : {fps_onnx:.1f} FPS")

    _print_table(artifacts, fps_onnx)
    return artifacts


def _benchmark_onnx(onnx_path: str, imgsz: int, n_frames: int) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: dummy})

    start = time.perf_counter()
    for _ in range(n_frames):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    return n_frames / elapsed


def _print_table(artifacts, fps_onnx):
    print("\n" + "=" * 50)
    print("  Export Summary")
    print("=" * 50)
    for key, path in artifacts.items():
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  {key:<16} {Path(path).name:<25} {size_mb:.1f} MB")
    print("-" * 50)
    print(f"  ONNX CPU FPS   : {fps_onnx:.1f}  (target: 5–8)")
    print(f"  VPU FPS target : ~25  (measure on device)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX/OpenVINO/blob")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--shaves", type=int, default=6, help="SHAVE cores for blob")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-blob", action="store_true")
    args = parser.parse_args()

    export(args.model, args.output, args.shaves, args.imgsz, skip_blob=args.skip_blob)
