#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path.home() / "bop_datasets" / "ycbv_3dsgrasp_eval"
META_PATH = ROOT / "metadata.csv"
OUT_CSV = ROOT / "metrics_3dsgrasp.csv"
OUT_BY_OBJ = ROOT / "metrics_by_object_3dsgrasp.csv"
OUT_SUMMARY = ROOT / "summary_3dsgrasp.txt"


def read_xyz(path):
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts[:, :3]


def read_pcd(path):
    """
    Reads simple PCD files with x y z fields.
    Supports ascii and binary PCD.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD, no DATA line: {path}")
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
            raise ValueError(f"Unsupported PCD DATA type {data_type}: {path}")

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

        pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
        return pts.astype(np.float32)


def normalize_by_partial(partial, *clouds):
    center = partial.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(partial - center, axis=1).max()
    if scale <= 0:
        scale = 1.0
    out = [(cloud - center) / scale for cloud in clouds]
    return out, center.squeeze(), float(scale)


def chamfer_and_fscore(pred, gt, thresholds=(0.01, 0.02)):
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    d_pred_to_gt, _ = tree_gt.query(pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1)

    cd_l1 = float(d_pred_to_gt.mean() + d_gt_to_pred.mean())
    cd_l2 = float((d_pred_to_gt ** 2).mean() + (d_gt_to_pred ** 2).mean())

    metrics = {
        "cd_l1": cd_l1,
        "cd_l2": cd_l2,
        "cd_l2_x1000": cd_l2 * 1000.0,
    }

    for t in thresholds:
        precision = float((d_pred_to_gt < t).mean())
        recall = float((d_gt_to_pred < t).mean())
        f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        key = str(t).replace(".", "p")
        metrics[f"precision_{key}"] = precision
        metrics[f"recall_{key}"] = recall
        metrics[f"fscore_{key}"] = float(f)

    return metrics


def bbox_diag(points):
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def main():
    meta = pd.read_csv(META_PATH)
    rows = []

    for _, r in meta.iterrows():
        name = r["sample_name"]

        partial_path = Path(r["partial_path"])
        gt_path = Path(r["gt_path"])
        comp_path = ROOT / "completions" / name / "complete_pc.pcd"
        comp_x_path = ROOT / "completions" / name / "complete_pc_x.pcd"

        if not comp_path.exists():
            print(f"[WARN] Missing completion: {name}")
            continue

        partial = read_xyz(partial_path)
        gt = read_xyz(gt_path)
        comp = read_pcd(comp_path)

        (partial_n, gt_n, comp_n), center, scale = normalize_by_partial(partial, partial, gt, comp)

        row = {
            "sample_name": name,
            "scene_id": int(r["scene_id"]),
            "im_id": int(r["im_id"]),
            "gt_idx": int(r["gt_idx"]),
            "obj_id": int(r["obj_id"]),
            "partial_points": len(partial),
            "gt_points": len(gt),
            "completion_points": len(comp),
            "normalization_scale": scale,
            "bbox_diag_partial_raw": bbox_diag(partial),
            "bbox_diag_gt_raw": bbox_diag(gt),
            "bbox_diag_completion_raw": bbox_diag(comp),
        }

        m_partial = chamfer_and_fscore(partial_n, gt_n)
        m_comp = chamfer_and_fscore(comp_n, gt_n)

        for k, v in m_partial.items():
            row[f"partial_{k}"] = v
        for k, v in m_comp.items():
            row[f"completed_{k}"] = v

        row["cd_l2_improvement_abs"] = row["partial_cd_l2"] - row["completed_cd_l2"]
        row["cd_l2_improvement_percent"] = 100.0 * row["cd_l2_improvement_abs"] / row["partial_cd_l2"]

        if comp_x_path.exists():
            comp_x = read_pcd(comp_x_path)
            (comp_x_n,), _, _ = normalize_by_partial(partial, comp_x)
            m_comp_x = chamfer_and_fscore(comp_x_n, gt_n)
            for k, v in m_comp_x.items():
                row[f"completed_x_{k}"] = v
            row["completed_x_cd_l2_improvement_percent"] = (
                100.0 * (row["partial_cd_l2"] - row["completed_x_cd_l2"]) / row["partial_cd_l2"]
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    by_obj = df.groupby("obj_id").agg(
        n=("sample_name", "count"),
        partial_cd_l2_x1000_mean=("partial_cd_l2_x1000", "mean"),
        completed_cd_l2_x1000_mean=("completed_cd_l2_x1000", "mean"),
        improvement_percent_mean=("cd_l2_improvement_percent", "mean"),
        partial_fscore_0p01_mean=("partial_fscore_0p01", "mean"),
        completed_fscore_0p01_mean=("completed_fscore_0p01", "mean"),
        partial_fscore_0p02_mean=("partial_fscore_0p02", "mean"),
        completed_fscore_0p02_mean=("completed_fscore_0p02", "mean"),
    ).reset_index()

    by_obj.to_csv(OUT_BY_OBJ, index=False)

    lines = []
    lines.append("3DSGrasp on YCB-V one-image-per-scene subset")
    lines.append("=" * 60)
    lines.append(f"Samples evaluated: {len(df)}")
    lines.append("")
    lines.append(f"Mean partial CD-L2 x1000:   {df['partial_cd_l2_x1000'].mean():.4f}")
    lines.append(f"Mean completed CD-L2 x1000: {df['completed_cd_l2_x1000'].mean():.4f}")
    lines.append(f"Mean improvement:           {df['cd_l2_improvement_percent'].mean():.2f}%")
    lines.append("")
    lines.append(f"Mean partial F-score@0.01:   {df['partial_fscore_0p01'].mean():.4f}")
    lines.append(f"Mean completed F-score@0.01: {df['completed_fscore_0p01'].mean():.4f}")
    lines.append(f"Mean partial F-score@0.02:   {df['partial_fscore_0p02'].mean():.4f}")
    lines.append(f"Mean completed F-score@0.02: {df['completed_fscore_0p02'].mean():.4f}")
    lines.append("")
    lines.append("Worst completed CD-L2 x1000 samples:")
    worst = df.sort_values("completed_cd_l2_x1000", ascending=False).head(10)
    for _, r in worst.iterrows():
        lines.append(
            f"{r['sample_name']}: completed={r['completed_cd_l2_x1000']:.4f}, "
            f"partial={r['partial_cd_l2_x1000']:.4f}, improvement={r['cd_l2_improvement_percent']:.2f}%"
        )

    OUT_SUMMARY.write_text("\n".join(lines))

    print("\n".join(lines))
    print()
    print(f"Saved per-sample metrics: {OUT_CSV}")
    print(f"Saved by-object metrics:  {OUT_BY_OBJ}")
    print(f"Saved summary:            {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
