"""P2 ReID evaluation: CMC curve, mAP, t-SNE, top-5 retrieval grid.

Protocol:
  - Split all crops into gallery (first half per identity) and query (second half)
  - For each query: rank gallery by embedding distance, check if correct identity is found
  - CMC@k: fraction of queries where correct match is in top-k results
  - mAP: mean Average Precision across all queries

Run:
    python -m oakd_vision.tracker.evaluate_reid
    python -m oakd_vision.tracker.evaluate_reid --model runs/reid/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from oakd_vision.tracker.reid_model import ReIDNet
from oakd_vision.tracker.triplet_dataset import ReIDDataset, build_transforms
from PIL import Image


def extract_embeddings(model, dataset, device):
    """Run all images through the model and return embeddings + labels."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    all_embs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            embs = model(imgs.to(device))
            all_embs.append(embs.cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_embs), np.array(all_labels)


def split_gallery_query(dataset):
    """Split dataset into gallery (first half per identity) and query (second half)."""
    from collections import defaultdict
    id_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        id_to_indices[label].append(idx)

    gallery_idx, query_idx = [], []
    for indices in id_to_indices.values():
        mid = max(1, len(indices) // 2)
        gallery_idx.extend(indices[:mid])
        query_idx.extend(indices[mid:])
    return gallery_idx, query_idx


def compute_cmc_map(query_embs, query_labels, gallery_embs, gallery_labels, max_rank=10):
    """Compute CMC curve and mAP."""
    # Pairwise distances: [Q, G]
    dist = torch.cdist(query_embs, gallery_embs).numpy()

    num_query = len(query_labels)
    cmc = np.zeros(max_rank)
    ap_sum = 0.0

    for q in range(num_query):
        q_label = query_labels[q]
        # Sort gallery by distance
        order = np.argsort(dist[q])
        matches = (gallery_labels[order] == q_label)

        # CMC: did correct match appear in top-k?
        for k in range(max_rank):
            if matches[:k+1].any():
                cmc[k:] += 1
                break

        # AP: average precision for this query
        num_relevant = matches.sum()
        if num_relevant == 0:
            continue
        hits = np.cumsum(matches)
        precision_at_k = hits / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / num_relevant
        ap_sum += ap

    cmc = cmc / num_query
    mAP = ap_sum / num_query
    return cmc, mAP


def plot_tsne(embeddings, labels, label_to_name, save_path):
    """2D t-SNE plot of all embeddings coloured by identity."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("  sklearn/matplotlib not installed — skipping t-SNE")
        return

    print("  Running t-SNE (may take 30s)...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings.numpy())

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_to_name.get(label, str(label))
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    c=[cmap(i)], label=name, s=30, alpha=0.8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.title("ReID Embedding Space (t-SNE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  t-SNE saved → {save_path}")


def plot_retrieval_grid(query_embs, query_labels, gallery_embs, gallery_labels,
                        query_dataset, gallery_dataset, label_to_name, save_path, n=5):
    """Show n query images with their top-5 gallery matches (green=correct, red=wrong)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("  matplotlib not installed — skipping retrieval grid")
        return

    dist = torch.cdist(query_embs, gallery_embs).numpy()
    # Pick n evenly-spaced queries
    query_indices = np.linspace(0, len(query_labels) - 1, n, dtype=int)

    fig, axes = plt.subplots(n, 6, figsize=(14, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis]

    for row, q_idx in enumerate(query_indices):
        q_label = query_labels[q_idx]
        order = np.argsort(dist[q_idx])[:5]

        # Query image
        q_path = query_dataset.samples[q_idx][0]
        q_img = np.array(Image.open(q_path).convert("RGB"))
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title(f"Query\n{label_to_name.get(q_label, q_label)}", fontsize=7)
        axes[row, 0].axis("off")
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(3)

        # Top-5 gallery matches
        for col, g_idx in enumerate(order):
            g_label = gallery_labels[g_idx]
            g_path = gallery_dataset.samples[g_idx][0]
            g_img = np.array(Image.open(g_path).convert("RGB"))
            color = "green" if g_label == q_label else "red"
            axes[row, col + 1].imshow(g_img)
            axes[row, col + 1].set_title(
                f"#{col+1} {label_to_name.get(g_label, g_label)}", fontsize=7)
            axes[row, col + 1].axis("off")
            for spine in axes[row, col + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    plt.suptitle("Top-5 Retrieval: Blue=Query | Green=Correct | Red=Wrong", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Retrieval grid saved → {save_path}")


def main(model_path: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = tuple(cfg["data"]["crop_size"])
    reid_dir = cfg["data"]["reid_dir"]
    output_dir = Path(cfg["training"]["checkpoint_dir"])

    print(f"Loading model from {model_path}...")
    model = ReIDNet(
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Full dataset (no augmentation for eval)
    full_dataset = ReIDDataset(reid_dir, crop_size=crop_size, augment=False)
    print(f"Total identities: {full_dataset.num_identities}  |  Total crops: {len(full_dataset)}")

    # Split into gallery and query
    gallery_idx, query_idx = split_gallery_query(full_dataset)

    from torch.utils.data import Subset
    gallery_dataset = Subset(full_dataset, gallery_idx)
    query_dataset   = Subset(full_dataset, query_idx)

    # Patch .samples so plot functions can access paths
    gallery_dataset.samples = [full_dataset.samples[i] for i in gallery_idx]
    query_dataset.samples   = [full_dataset.samples[i] for i in query_idx]

    print(f"Gallery: {len(gallery_idx)} crops  |  Query: {len(query_idx)} crops")

    # Extract embeddings
    print("Extracting gallery embeddings...")
    gallery_embs, gallery_labels = extract_embeddings(model, gallery_dataset, device)
    print("Extracting query embeddings...")
    query_embs, query_labels = extract_embeddings(model, query_dataset, device)

    # CMC + mAP
    print("\nComputing CMC and mAP...")
    cmc, mAP = compute_cmc_map(query_embs, query_labels, gallery_embs, gallery_labels)

    print("\n--- Results ---")
    print(f"  Rank-1:  {cmc[0]:.3f}  (target ≥ 0.70)")
    print(f"  Rank-3:  {cmc[2]:.3f}")
    print(f"  Rank-5:  {cmc[4]:.3f}")
    print(f"  Rank-10: {cmc[9]:.3f}")
    print(f"  mAP:     {mAP:.3f}  (target ≥ 0.50)")

    # Plots
    all_embs = torch.cat([gallery_embs, query_embs])
    all_labels = np.concatenate([gallery_labels, query_labels])

    print("\nGenerating plots...")
    plot_tsne(all_embs, all_labels, full_dataset.label_to_name,
              save_path=output_dir / "tsne.png")
    plot_retrieval_grid(query_embs, query_labels, gallery_embs, gallery_labels,
                        query_dataset, gallery_dataset, full_dataset.label_to_name,
                        save_path=output_dir / "retrieval_grid.png")

    # Save report
    report = output_dir / "evaluation_report.md"
    with open(report, "w") as f:
        f.write("# P2 ReID Evaluation Report\n\n")
        f.write(f"**Model:** `{model_path}`\n\n")
        f.write(f"| Metric | Score | Target |\n|---|---|---|\n")
        f.write(f"| Rank-1 | {cmc[0]:.3f} | ≥ 0.70 |\n")
        f.write(f"| Rank-3 | {cmc[2]:.3f} | — |\n")
        f.write(f"| Rank-5 | {cmc[4]:.3f} | — |\n")
        f.write(f"| mAP | {mAP:.3f} | ≥ 0.50 |\n\n")
        f.write("![t-SNE](tsne.png)\n\n")
        f.write("![Retrieval Grid](retrieval_grid.png)\n")
    print(f"\nReport saved → {report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/reid/best.pt")
    parser.add_argument("--config", default="training/configs/reid_config.yaml")
    args = parser.parse_args()
    main(args.model, args.config)
