#!/usr/bin/env python3
from pathlib import Path
import importlib.util

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path.home() / "bop_datasets" / "ycbv_3dsgrasp_eval_filtered"
YCBV_ROOT = Path.home() / "bop_datasets" / "ycbv"
METRICS = ROOT / "metrics_3dsgrasp.csv"
OUT_DIR = ROOT / "visualizations_with_rgb"

# Re-use helper functions from the previous visualization script
HELPER_PATH = Path.home() / "shape_completion_eval" / "visualize_3dsgrasp_results.py"
spec = importlib.util.spec_from_file_location("vis_helpers", HELPER_PATH)
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def crop_rgb_with_mask(scene_id, im_id, gt_idx, pad=20):
    scene_path = YCBV_ROOT / "test" / f"{scene_id:06d}"
    rgb_path = scene_path / "rgb" / f"{im_id:06d}.png"
    mask_path = scene_path / "mask_visib" / f"{im_id:06d}_{gt_idx:06d}.png"

    rgb = imageio.imread(rgb_path)
    mask = imageio.imread(mask_path) > 0

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(rgb.shape[0] - 1, y1 + pad)
    x1 = min(rgb.shape[1] - 1, x1 + pad)

    crop = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1]

    overlay = crop.astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255

    alpha = 0.35
    overlay[crop_mask] = (1 - alpha) * overlay[crop_mask] + alpha * red[crop_mask]

    return overlay.astype(np.uint8)


def plot_sample(row, label, rank):
    name = row["sample_name"]
    scene_id = int(row["scene_id"])
    im_id = int(row["im_id"])
    gt_idx = int(row["gt_idx"])

    partial_path = ROOT / "partials" / f"{name}.xyz"
    gt_path = ROOT / "gt" / f"{name}.xyz"
    comp_path = ROOT / "completions" / name / "complete_pc.pcd"

    rgb_crop = crop_rgb_with_mask(scene_id, im_id, gt_idx)

    partial = vis.read_xyz(partial_path)
    gt = vis.read_xyz(gt_path)
    comp = vis.read_pcd(comp_path)

    partial_n, comp_n, gt_n = vis.normalize_by_partial(partial, partial, comp, gt)

    partial_n = vis.sample_points(partial_n, seed=42)
    comp_n = vis.sample_points(comp_n, seed=43)
    gt_n = vis.sample_points(gt_n, seed=44)

    all_points = np.vstack([partial_n, comp_n, gt_n])

    fig = plt.figure(figsize=(18, 5))

    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(rgb_crop)
    ax0.set_title("RGB crop + mask")
    ax0.axis("off")

    clouds = [
        ("Partial input", partial_n),
        ("3DSGrasp completion", comp_n),
        ("GT object model", gt_n),
    ]

    for i, (title, pts) in enumerate(clouds, start=2):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(title)
        vis.set_equal_axes(ax, all_points)
        ax.view_init(elev=20, azim=-60)

    fig.suptitle(
        f"{label.upper()} #{rank}: {name}\n"
        f"obj_id={int(row['obj_id'])} | "
        f"partial CD-L2x1000={row['partial_cd_l2_x1000']:.2f} | "
        f"completed CD-L2x1000={row['completed_cd_l2_x1000']:.2f} | "
        f"improvement={row['cd_l2_improvement_percent']:.1f}%"
    )

    plt.tight_layout()

    out_path = OUT_DIR / f"{label}_{rank:02d}_{name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS)

    best = df.sort_values("cd_l2_improvement_percent", ascending=False).head(4)
    worst = df.sort_values("cd_l2_improvement_percent", ascending=True).head(4)

    median_imp = df["cd_l2_improvement_percent"].median()
    typical = df.assign(
        dist_to_median=(df["cd_l2_improvement_percent"] - median_imp).abs()
    ).sort_values("dist_to_median").head(4)

    rows = []

    for label, subset in [("best", best), ("worst", worst), ("typical", typical)]:
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            out_path = plot_sample(row, label, rank)
            rows.append({
                "label": label,
                "rank": rank,
                "sample_name": row["sample_name"],
                "obj_id": int(row["obj_id"]),
                "image_path": str(out_path),
            })
            print(f"Saved {out_path}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "selected_visualizations_with_rgb.csv", index=False)
    print()
    print(f"Visualization folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
