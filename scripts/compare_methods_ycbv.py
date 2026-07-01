from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import struct

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()

PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
G3D_DIR = ROOT / "completions"
ADA_DIR = ROOT / "adapointr_completions_norm"

OUT_DIR = ROOT / "analysis_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_xyz(path):
    pts = np.loadtxt(path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    return pts[:, :3]


def read_npy(path):
    pts = np.load(path).astype(np.float32)
    pts = np.squeeze(pts)
    return pts[:, :3]


def read_pcd(path):
    """
    Robust PCD reader for our 3DSGrasp outputs.
    Handles:
    - normal ascii PCD with one point per row
    - ascii PCD where all xyz values are stored in one long row
    - simple binary PCD with x/y/z fields
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            line_s = line.decode("utf-8", errors="ignore").strip()
            header.append(line_s)
            if line_s.startswith("DATA"):
                data_type = line_s.split()[1]
                break

        info = {}
        for line in header:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0]] = parts[1:]

        fields = info.get("FIELDS", [])
        n_points = int(info.get("POINTS", info.get("WIDTH", ["0"]))[0])

        if data_type == "ascii":
            txt = f.read().decode("utf-8", errors="ignore")
            vals = np.fromstring(txt, sep=" ", dtype=np.float32)

            # Case 1: exactly xyzxyzxyz...
            if len(fields) == 3 and set(fields) >= {"x", "y", "z"}:
                if vals.size % 3 != 0:
                    raise ValueError(f"ASCII PCD has {vals.size} values, not divisible by 3: {path}")
                return vals.reshape(-1, 3)

            # Case 2: more fields, use header layout
            if n_points > 0 and len(fields) > 0:
                cols_per_point = len(fields)
                arr = vals.reshape(-1, cols_per_point)
                cols = {name: i for i, name in enumerate(fields)}
                return arr[:, [cols["x"], cols["y"], cols["z"]]].astype(np.float32)

            # Fallback
            if vals.size % 3 == 0:
                return vals.reshape(-1, 3)

            raise ValueError(f"Could not parse ASCII PCD: {path}, values={vals.size}, fields={fields}")

        if data_type == "binary":
            size = list(map(int, info.get("SIZE", [])))
            typ = info.get("TYPE", [])
            count = list(map(int, info.get("COUNT", ["1"] * len(fields))))

            dtype_fields = []
            for name, sz, tp, cnt in zip(fields, size, typ, count):
                if tp == "F" and sz == 4:
                    dt = np.float32
                elif tp == "F" and sz == 8:
                    dt = np.float64
                elif tp == "U" and sz == 4:
                    dt = np.uint32
                elif tp == "I" and sz == 4:
                    dt = np.int32
                else:
                    dt = np.float32
                dtype_fields.append((name, dt, cnt))

            dtype = np.dtype(dtype_fields)
            arr = np.frombuffer(f.read(), dtype=dtype, count=n_points)
            return np.vstack([arr["x"], arr["y"], arr["z"]]).T.astype(np.float32)

        raise ValueError(f"Unsupported PCD DATA type: {data_type}")

def chamfer(pred, gt):
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)
    d1, _ = tree_gt.query(pred, k=1)
    d2, _ = tree_pred.query(gt, k=1)
    return float(((d1 ** 2).mean() + (d2 ** 2).mean()) * 1000.0)


def main():
    rows = []

    for partial_path in sorted(PARTIAL_DIR.glob("*.xyz")):
        name = partial_path.stem

        gt_path = GT_DIR / f"{name}.xyz"
        g3d_path = G3D_DIR / name / "complete_pc.pcd"
        ada_path = ADA_DIR / name / "fine.npy"

        if not gt_path.exists() or not g3d_path.exists() or not ada_path.exists():
            print("Skipping missing:", name)
            continue

        partial = read_xyz(partial_path)
        gt = read_xyz(gt_path)
        g3d = read_pcd(g3d_path)

        # Fix for 3DSGrasp PCDs that are read as one long row:
        # shape (1, 24576) -> (8192, 3)
        g3d = np.asarray(g3d, dtype=np.float32)
        if g3d.ndim == 1:
            if g3d.size % 3 != 0:
                raise ValueError(f"3DSGrasp output cannot be reshaped: {g3d.shape}")
            g3d = g3d.reshape(-1, 3)
        if g3d.ndim == 2 and g3d.shape[0] == 1 and g3d.shape[1] % 3 == 0:
            g3d = g3d.reshape(-1, 3)
        if g3d.ndim != 2 or g3d.shape[1] != 3:
            raise ValueError(f"Bad 3DSGrasp point shape after fix: {g3d.shape}")

        ada = read_npy(ada_path)

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale
        g3d_n = (g3d - center[None, :]) / scale

        partial_cd = chamfer(partial_n, gt_n)
        g3d_cd = chamfer(g3d_n, gt_n)
        ada_cd = chamfer(ada, gt_n)

        obj_id = int(name.split("_obj")[-1])

        rows.append({
            "name": name,
            "obj_id": obj_id,
            "partial_cd_l2_x1000": partial_cd,
            "3dsgrasp_cd_l2_x1000": g3d_cd,
            "adapointr_cd_l2_x1000": ada_cd,
            "3dsgrasp_improvement_percent": (partial_cd - g3d_cd) / partial_cd * 100,
            "adapointr_improvement_percent": (partial_cd - ada_cd) / partial_cd * 100,
            "winner": "3DSGrasp" if g3d_cd < ada_cd else "AdaPoinTr",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "compare_3dsgrasp_adapointr.csv", index=False)

    summary = []
    summary.append("3DSGrasp vs AdaPoinTr on filtered YCB-V partial point clouds")
    summary.append("=" * 70)
    summary.append(f"Samples: {len(df)}")
    summary.append("")
    summary.append(f"Mean Partial CD-L2 x1000:   {df['partial_cd_l2_x1000'].mean():.4f}")
    summary.append(f"Mean 3DSGrasp CD-L2 x1000:  {df['3dsgrasp_cd_l2_x1000'].mean():.4f}")
    summary.append(f"Mean AdaPoinTr CD-L2 x1000: {df['adapointr_cd_l2_x1000'].mean():.4f}")
    summary.append("")
    summary.append(f"3DSGrasp improved:  {(df['3dsgrasp_improvement_percent'] > 0).sum()} / {len(df)}")
    summary.append(f"AdaPoinTr improved: {(df['adapointr_improvement_percent'] > 0).sum()} / {len(df)}")
    summary.append("")
    summary.append("Wins:")
    summary.append(df["winner"].value_counts().to_string())
    summary.append("")
    summary.append("Best 3DSGrasp over AdaPoinTr:")
    tmp = df.assign(delta=df["adapointr_cd_l2_x1000"] - df["3dsgrasp_cd_l2_x1000"])
    summary.append(tmp.sort_values("delta", ascending=False).head(5).to_string(index=False))

    (OUT_DIR / "summary_compare.txt").write_text("\n".join(summary))
    print("\n".join(summary))

    # Bar plot means
    means = [
        df["partial_cd_l2_x1000"].mean(),
        df["3dsgrasp_cd_l2_x1000"].mean(),
        df["adapointr_cd_l2_x1000"].mean(),
    ]
    labels = ["Partial", "3DSGrasp", "AdaPoinTr"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, means)
    plt.ylabel("CD-L2 x1000 lower is better")
    plt.title("Mean shape completion error")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_cd_l2_bar.png", dpi=200)
    plt.close()

    # Per sample sorted plot
    sdf = df.sort_values("partial_cd_l2_x1000").reset_index(drop=True)
    plt.figure(figsize=(10, 4))
    plt.plot(sdf.index, sdf["partial_cd_l2_x1000"], label="Partial")
    plt.plot(sdf.index, sdf["3dsgrasp_cd_l2_x1000"], label="3DSGrasp")
    plt.plot(sdf.index, sdf["adapointr_cd_l2_x1000"], label="AdaPoinTr")
    plt.ylabel("CD-L2 x1000 lower is better")
    plt.xlabel("Samples sorted by partial error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_sample_cd_l2.png", dpi=200)
    plt.close()

    # Scatter 3DSGrasp vs AdaPoinTr
    plt.figure(figsize=(5, 5))
    plt.scatter(df["3dsgrasp_cd_l2_x1000"], df["adapointr_cd_l2_x1000"], s=18)
    lim = max(df["3dsgrasp_cd_l2_x1000"].max(), df["adapointr_cd_l2_x1000"].max())
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("3DSGrasp CD-L2 x1000")
    plt.ylabel("AdaPoinTr CD-L2 x1000")
    plt.title("Below diagonal: AdaPoinTr better")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_3dsgrasp_vs_adapointr.png", dpi=200)
    plt.close()

    print("")
    print("Wrote analysis to:", OUT_DIR)


if __name__ == "__main__":
    main()
