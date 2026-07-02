from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
PARTIAL_DIR = ROOT / "partials"
CROP_DIR = ROOT / "vlm_object_crops"

REASONING_DIR = Path("results/vlm_reasoning")
OUT_DIR = Path("results/task_grasp_filter_demo")
VIS_DIR = OUT_DIR / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

OBJ_NAME_TO_ID = {
    "mug": 14,
    "mustard bottle": 5,
    "power drill": 15,
    "banana": 10,
    "cracker box": 2,
    "tomato soup can": 4,
}

def force_xyz(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.squeeze(x)
    if x.ndim == 1:
        return x.reshape(-1, 3)
    if x.ndim == 2 and x.shape[1] == 3:
        return x
    return x.reshape(-1, 3)

def read_xyz(path):
    return force_xyz(np.loadtxt(path))

def parse_obj_id(name):
    m = re.search(r"_obj(\d+)", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1))

def normalize_cloud(pts):
    pts = force_xyz(pts)
    center = pts.mean(axis=0)
    scale = np.linalg.norm(pts - center[None, :], axis=1).max()
    scale = max(float(scale), 1e-6)
    return (pts - center[None, :]) / scale

def sample_candidates(pts, n=350):
    rng = np.random.default_rng(0)
    pts = force_xyz(pts)
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]

def score_candidates(candidates, rule):
    pts = force_xyz(candidates)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    z_norm = (z - z.min()) / max(float(z.max() - z.min()), 1e-6)

    xy = pts[:, :2]
    xy_center = xy.mean(axis=0)
    radial = np.linalg.norm(xy - xy_center[None, :], axis=1)
    radial_norm = radial / max(float(radial.max()), 1e-6)

    mid_z = 1.0 - np.abs(z_norm - 0.5) * 2.0
    mid_z = np.clip(mid_z, 0.0, 1.0)

    # Baseline: generic geometry-only score.
    # This simulates "take a generally good central grasp".
    baseline_score = 0.8 * mid_z + 0.2 * radial_norm

    z_range = rule.get("preferred_z_range", [0.0, 1.0])
    if not isinstance(z_range, list) or len(z_range) != 2:
        z_range = [0.0, 1.0]

    lo = float(z_range[0])
    hi = float(z_range[1])
    lo = max(0.0, min(1.0, lo))
    hi = max(0.0, min(1.0, hi))

    z_center = (lo + hi) / 2.0
    z_width = max((hi - lo) / 2.0, 0.05)

    z_preference = 1.0 - np.abs(z_norm - z_center) / z_width
    z_preference = np.clip(z_preference, 0.0, 1.0)

    task_score = z_preference.copy()

    if rule.get("avoid_top", False):
        task_score -= np.clip((z_norm - 0.75) / 0.25, 0.0, 1.0)

    if rule.get("avoid_bottom", False):
        task_score -= np.clip((0.25 - z_norm) / 0.25, 0.0, 1.0)

    if rule.get("prefer_side", False):
        task_score += 0.8 * radial_norm

    if rule.get("prefer_middle_body", False):
        task_score += 0.8 * mid_z

    if rule.get("prefer_handle", False):
        # Approximation: handle is treated as side/protruding geometry.
        task_score += 1.0 * radial_norm

    if rule.get("prefer_edge_or_corner", False):
        edge = np.maximum(np.abs(x), np.abs(y))
        edge = edge / max(float(edge.max()), 1e-6)
        task_score += 0.8 * edge

    final_score = 0.5 * baseline_score + task_score

    valid = (z_norm >= lo) & (z_norm <= hi)

    if rule.get("avoid_top", False):
        valid &= z_norm < 0.80

    if rule.get("avoid_bottom", False):
        valid &= z_norm > 0.20

    if rule.get("prefer_side", False) or rule.get("prefer_handle", False):
        valid &= radial_norm > np.quantile(radial_norm, 0.55)

    if rule.get("prefer_middle_body", False):
        valid &= mid_z > 0.40

    return baseline_score, task_score, final_score, valid, z_norm, radial_norm

def find_crop(sample_name):
    p = CROP_DIR / f"{sample_name}.png"
    if p.exists():
        return p
    return None

def plot_case(sample_name, obj_name, task, pts, candidates, baseline_idx, filtered_idx, valid, out_path):
    crop = find_crop(sample_name)

    fig = plt.figure(figsize=(15, 5))

    ax0 = fig.add_subplot(1, 3, 1)
    if crop is not None:
        ax0.imshow(Image.open(crop).convert("RGB"))
    ax0.set_title("RGB crop + mask")
    ax0.axis("off")

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.15)
    ax1.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], s=8, alpha=0.30)
    ax1.scatter(
        candidates[baseline_idx, 0],
        candidates[baseline_idx, 1],
        candidates[baseline_idx, 2],
        s=120,
        marker="x",
    )
    ax1.set_title("Baseline selected")

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.15)
    ax2.scatter(candidates[valid, 0], candidates[valid, 1], candidates[valid, 2], s=8, alpha=0.40)
    ax2.scatter(
        candidates[filtered_idx, 0],
        candidates[filtered_idx, 1],
        candidates[filtered_idx, 2],
        s=120,
        marker="x",
    )
    ax2.set_title("VLM-filtered selected")

    for ax in [ax1, ax2]:
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        c = (mn + mx) / 2.0
        r = (mx - mn).max() / 2.0
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=45)

    fig.suptitle(f"{obj_name}: {task}\n{sample_name}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    rows = []

    for reasoning_path in sorted(REASONING_DIR.glob("*.json")):
        data = json.loads(reasoning_path.read_text())

        obj_name = str(data.get("object", "")).lower().strip()
        task = str(data.get("task", "")).strip()
        is_grasp_task = bool(data.get("is_grasp_task", True))
        rule = data.get("simple_rule", {})

        if not is_grasp_task:
            print("Skipping non-grasp task:", reasoning_path.name)
            continue

        if obj_name not in OBJ_NAME_TO_ID:
            print("Skipping unknown object:", obj_name)
            continue

        obj_id = OBJ_NAME_TO_ID[obj_name]
        sample_paths = [
            p for p in sorted(PARTIAL_DIR.glob("*.xyz"))
            if parse_obj_id(p.stem) == obj_id
        ]

        if not sample_paths:
            print("No samples found for:", obj_name, obj_id)
            continue

        for sample_idx, partial_path in enumerate(sample_paths):
            sample_name = partial_path.stem

            pts = normalize_cloud(read_xyz(partial_path))
            candidates = sample_candidates(pts, n=350)

            baseline_score, task_score, final_score, valid, z_norm, radial_norm = score_candidates(candidates, rule)

            baseline_idx = int(np.argmax(baseline_score))
            filtered_idx = int(np.argmax(final_score))

            rows.append({
                "reasoning_file": reasoning_path.name,
                "sample_name": sample_name,
                "object": obj_name,
                "task": task,
                "preferred_grasp_region": data.get("preferred_grasp_region"),
                "avoid_grasp_region": data.get("avoid_grasp_region"),
                "baseline_valid": bool(valid[baseline_idx]),
                "filtered_valid": bool(valid[filtered_idx]),
                "num_candidates": int(len(candidates)),
                "num_valid_candidates": int(valid.sum()),
                "valid_candidate_fraction": float(valid.mean()),
                "baseline_z_norm": float(z_norm[baseline_idx]),
                "filtered_z_norm": float(z_norm[filtered_idx]),
                "baseline_radial_norm": float(radial_norm[baseline_idx]),
                "filtered_radial_norm": float(radial_norm[filtered_idx]),
            })

            if sample_idx == 0:
                out_png = VIS_DIR / f"{reasoning_path.stem}_{sample_name}.png"
                plot_case(
                    sample_name,
                    obj_name,
                    task,
                    pts,
                    candidates,
                    baseline_idx,
                    filtered_idx,
                    valid,
                    out_png,
                )

    df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "task_grasp_filter_metrics.csv"
    summary_path = OUT_DIR / "task_grasp_filter_summary.txt"

    df.to_csv(metrics_path, index=False)

    lines = []
    lines.append("Offline VLM-based task-conditioned grasp filter demo")
    lines.append("=" * 90)
    lines.append("")
    lines.append("This is a proof-of-concept without robot execution.")
    lines.append("It uses sampled point-cloud candidates instead of real GraspGen proposals.")
    lines.append("The same filter can later be applied to real GraspGen candidate grasps.")
    lines.append("")
    lines.append(f"Evaluated candidates from samples: {len(df)}")
    lines.append("")

    if len(df) > 0:
        lines.append(f"Baseline Top-1 task-valid rate:     {df['baseline_valid'].mean() * 100:.2f}%")
        lines.append(f"VLM-filtered Top-1 task-valid rate: {df['filtered_valid'].mean() * 100:.2f}%")
        lines.append("")
        lines.append("Per object valid Top-1 rate [%]:")
        grouped = df.groupby("object")[["baseline_valid", "filtered_valid"]].mean() * 100
        lines.append(grouped.to_string())
        lines.append("")

    lines.append("Files:")
    lines.append(f"- {metrics_path}")
    lines.append(f"- {summary_path}")
    lines.append(f"- {VIS_DIR}")

    summary = "\n".join(lines)
    summary_path.write_text(summary)
    print(summary)

if __name__ == "__main__":
    main()
