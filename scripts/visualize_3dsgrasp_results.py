#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path.home() / "bop_datasets" / "ycbv_3dsgrasp_eval_filtered"
METRICS = ROOT / "metrics_3dsgrasp.csv"
OUT_DIR = ROOT / "visualizations"

N_SHOW = 2048
RANDOM_SEED = 42


def read_xyz(path):
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :3]


def read_pcd(path):
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD: {path}")
            header_lines.append(line.decode("utf-8", errors="ignore").strip())
            if line.startswith(b"DATA"):
                data_start = f.tell()
                break

        header = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            header[parts[0].upper()] = parts[1:]

        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        n_points = int(header["POINTS"][0])
        data_type = header["DATA"][0].lower()

        if data_type == "ascii":
            arr = np.loadtxt(path, comments="#", skiprows=len(header_lines), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
            return arr[:, [ix, iy, iz]].astype(np.float32)

        if data_type != "binary":
            raise ValueError(f"Unsupported PCD DATA type: {data_type}")

        dtype_fields = []
        for field, size, typ, count in zip(fields, sizes, types, counts):
            if typ == "F" and size == 4:
                dt = np.float32
            elif typ == "F" and size == 8:
                dt = np.float64
            elif typ == "U" and size == 1:
                dt = np.uint8
            elif typ == "U" and size == 2:
                dt = np.uint16
            elif typ == "U" and size == 4:
                dt = np.uint32
            elif typ == "I" and size == 1:
                dt = np.int8
            elif typ == "I" and size == 2:
                dt = np.int16
            elif typ == "I" and size == 4:
                dt = np.int32
            else:
                raise ValueError(f"Unsupported PCD field type: {field} {typ}{size}")

            if count == 1:
                dtype_fields.append((field, dt))
            else:
                dtype_fields.append((field, dt, (count,)))

        dtype = np.dtype(dtype_fields)
        f.seek(data_start)
        raw = f.read(n_points * dtype.itemsize)
        arr = np.frombuffer(raw, dtype=dtype, count=n_points)

    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)


def sample_points(points, n=N_SHOW, seed=0):
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def normalize_by_partial(partial, *clouds):
    center = partial.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(partial - center, axis=1).max()
    if scale <= 0:
        scale = 1.0
    return [(c - center) / scale for c in clouds]


def set_equal_axes(ax, all_points):
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_sample(row, label, rank):
    name = row["sample_name"]

    partial_path = ROOT / "partials" / f"{name}.xyz"
    gt_path = ROOT / "gt" / f"{name}.xyz"
    comp_path = ROOT / "completions" / name / "complete_pc.pcd"

    partial = read_xyz(partial_path)
    gt = read_xyz(gt_path)
    comp = read_pcd(comp_path)

    partial_n, comp_n, gt_n = normalize_by_partial(partial, partial, comp, gt)

    partial_n = sample_points(partial_n, seed=RANDOM_SEED)
    comp_n = sample_points(comp_n, seed=RANDOM_SEED + 1)
    gt_n = sample_points(gt_n, seed=RANDOM_SEED + 2)

    all_points = np.vstack([partial_n, comp_n, gt_n])

    fig = plt.figure(figsize=(15, 5))

    clouds = [
        ("Partial input", partial_n),
        ("3DSGrasp completion", comp_n),
        ("GT object model", gt_n),
    ]

    for i, (title, pts) in enumerate(clouds, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(title)
        set_equal_axes(ax, all_points)
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

    # Typical samples: closest to median improvement.
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
                "partial_cd_l2_x1000": row["partial_cd_l2_x1000"],
                "completed_cd_l2_x1000": row["completed_cd_l2_x1000"],
                "improvement_percent": row["cd_l2_improvement_percent"],
                "image_path": str(out_path),
            })
            print(f"Saved {out_path}")

    selected_csv = OUT_DIR / "selected_visualizations.csv"
    pd.DataFrame(rows).to_csv(selected_csv, index=False)
    print()
    print(f"Saved selection table: {selected_csv}")
    print(f"Visualization folder:  {OUT_DIR}")


if __name__ == "__main__":
    main()
