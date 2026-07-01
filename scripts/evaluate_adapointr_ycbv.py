from pathlib import Path
import csv
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()

PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
ADAPOINTR_DIR = ROOT / "adapointr_completions_norm"
TRANSFORMS_CSV = ROOT / "adapointr_transforms.csv"

OUT_METRICS = ROOT / "metrics_adapointr.csv"
OUT_BY_OBJECT = ROOT / "metrics_by_object_adapointr.csv"
OUT_SUMMARY = ROOT / "summary_adapointr.txt"
OUT_COMPARE = ROOT / "compare_3dsgrasp_vs_adapointr.csv"


def read_xyz(path):
    pts = np.loadtxt(path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    return pts[:, :3]


def read_npy_points(path):
    pts = np.load(path).astype(np.float32)
    pts = np.squeeze(pts)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Unexpected point shape {pts.shape} in {path}")
    return pts


def chamfer_and_fscore(pred, gt, thresholds=(0.01, 0.02)):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    d_pred_to_gt, _ = tree_gt.query(pred, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1)

    cd_l1 = float(d_pred_to_gt.mean() + d_gt_to_pred.mean())
    cd_l2 = float((d_pred_to_gt ** 2).mean() + (d_gt_to_pred ** 2).mean())

    out = {
        "cd_l1": cd_l1,
        "cd_l2": cd_l2,
        "cd_l2_x1000": cd_l2 * 1000.0,
    }

    for th in thresholds:
        precision = float((d_pred_to_gt < th).mean())
        recall = float((d_gt_to_pred < th).mean())
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0

        key = str(th).replace(".", "p")
        out[f"precision_{key}"] = precision
        out[f"recall_{key}"] = recall
        out[f"fscore_{key}"] = fscore

    return out


def load_transforms():
    transforms = {}
    with open(TRANSFORMS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            center = np.array([
                float(row["center_x"]),
                float(row["center_y"]),
                float(row["center_z"]),
            ], dtype=np.float32)
            scale = float(row["scale"])
            transforms[name] = (center, scale)
    return transforms


def normalize_with_transform(pts, center, scale):
    return (pts - center[None, :]) / scale


def main():
    transforms = load_transforms()
    rows = []

    for partial_path in sorted(PARTIAL_DIR.glob("*.xyz")):
        name = partial_path.stem

        gt_path = GT_DIR / f"{name}.xyz"
        pred_path = ADAPOINTR_DIR / name / "fine.npy"

        if not gt_path.exists():
            print("Missing GT:", gt_path)
            continue

        if not pred_path.exists():
            print("Missing AdaPoinTr output:", pred_path)
            continue

        if name not in transforms:
            print("Missing transform:", name)
            continue

        center, scale = transforms[name]

        partial = read_xyz(partial_path)
        gt = read_xyz(gt_path)
        pred = read_npy_points(pred_path)

        partial_n = normalize_with_transform(partial, center, scale)
        gt_n = normalize_with_transform(gt, center, scale)

        partial_metrics = chamfer_and_fscore(partial_n, gt_n)
        pred_metrics = chamfer_and_fscore(pred, gt_n)

        obj_id = int(name.split("_obj")[-1])

        row = {
            "name": name,
            "obj_id": obj_id,
            "num_partial_points": len(partial),
            "num_gt_points": len(gt),
            "num_adapointr_points": len(pred),
            "partial_cd_l2_x1000": partial_metrics["cd_l2_x1000"],
            "adapointr_cd_l2_x1000": pred_metrics["cd_l2_x1000"],
            "improvement_percent": (
                (partial_metrics["cd_l2"] - pred_metrics["cd_l2"])
                / partial_metrics["cd_l2"] * 100.0
                if partial_metrics["cd_l2"] > 0 else np.nan
            ),
            "partial_fscore_0p01": partial_metrics["fscore_0p01"],
            "adapointr_fscore_0p01": pred_metrics["fscore_0p01"],
            "partial_fscore_0p02": partial_metrics["fscore_0p02"],
            "adapointr_fscore_0p02": pred_metrics["fscore_0p02"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_METRICS, index=False)

    by_obj = df.groupby("obj_id").agg({
        "name": "count",
        "partial_cd_l2_x1000": "mean",
        "adapointr_cd_l2_x1000": "mean",
        "improvement_percent": "mean",
        "partial_fscore_0p01": "mean",
        "adapointr_fscore_0p01": "mean",
        "partial_fscore_0p02": "mean",
        "adapointr_fscore_0p02": "mean",
    }).rename(columns={"name": "num_samples"}).reset_index()

    by_obj.to_csv(OUT_BY_OBJECT, index=False)

    improved = int((df["improvement_percent"] > 0).sum())
    n = len(df)

    lines = []
    lines.append("AdaPoinTr on filtered YCB-V partial point clouds")
    lines.append("=" * 55)
    lines.append(f"Samples evaluated: {n}")
    lines.append(f"Improved samples: {improved} / {n}")
    lines.append("")
    lines.append(f"Mean partial CD-L2 x1000:    {df['partial_cd_l2_x1000'].mean():.4f}")
    lines.append(f"Mean AdaPoinTr CD-L2 x1000:  {df['adapointr_cd_l2_x1000'].mean():.4f}")
    lines.append(f"Mean improvement:            {df['improvement_percent'].mean():.2f}%")
    lines.append(f"Median improvement:          {df['improvement_percent'].median():.2f}%")
    lines.append("")
    lines.append(f"Mean partial F-score@0.01:    {df['partial_fscore_0p01'].mean():.4f}")
    lines.append(f"Mean AdaPoinTr F-score@0.01:  {df['adapointr_fscore_0p01'].mean():.4f}")
    lines.append(f"Mean partial F-score@0.02:    {df['partial_fscore_0p02'].mean():.4f}")
    lines.append(f"Mean AdaPoinTr F-score@0.02:  {df['adapointr_fscore_0p02'].mean():.4f}")
    lines.append("")
    lines.append("Best samples:")
    lines.append(df.sort_values("improvement_percent", ascending=False).head(5).to_string(index=False))
    lines.append("")
    lines.append("Worst samples:")
    lines.append(df.sort_values("improvement_percent", ascending=True).head(5).to_string(index=False))

    OUT_SUMMARY.write_text("\n".join(lines))
    print("\n".join(lines))

    three_path = ROOT / "metrics_3dsgrasp.csv"
    if three_path.exists():
        g = pd.read_csv(three_path)
        merged = df.merge(
            g[["name", "completed_cd_l2_x1000", "improvement_percent"]],
            on="name",
            how="inner",
            suffixes=("_adapointr", "_3dsgrasp"),
        )
        merged = merged.rename(columns={
            "completed_cd_l2_x1000": "3dsgrasp_cd_l2_x1000",
            "adapointr_cd_l2_x1000": "adapointr_cd_l2_x1000",
            "improvement_percent_adapointr": "adapointr_improvement_percent",
            "improvement_percent_3dsgrasp": "3dsgrasp_improvement_percent",
        })
        merged["winner_cd_l2"] = np.where(
            merged["adapointr_cd_l2_x1000"] < merged["3dsgrasp_cd_l2_x1000"],
            "AdaPoinTr",
            "3DSGrasp",
        )
        merged.to_csv(OUT_COMPARE, index=False)

        print("")
        print("Comparison with 3DSGrasp")
        print("=" * 55)
        print(f"Mean AdaPoinTr CD-L2 x1000: {merged['adapointr_cd_l2_x1000'].mean():.4f}")
        print(f"Mean 3DSGrasp CD-L2 x1000:  {merged['3dsgrasp_cd_l2_x1000'].mean():.4f}")
        print("Wins:")
        print(merged["winner_cd_l2"].value_counts().to_string())
        print("")
        print("Wrote comparison to:", OUT_COMPARE)

    print("")
    print("Wrote:")
    print(" ", OUT_METRICS)
    print(" ", OUT_BY_OBJECT)
    print(" ", OUT_SUMMARY)


if __name__ == "__main__":
    main()
