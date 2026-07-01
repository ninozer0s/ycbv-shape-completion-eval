from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()

COMPARE = ROOT / "analysis_compare" / "compare_3dsgrasp_adapointr.csv"
PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
G3D_DIR = ROOT / "completions"
ADA_DIR = ROOT / "adapointr_completions_norm"

OUT_DIR = ROOT / "visualizations_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_xyz(path):
    pts = np.loadtxt(path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        pts = pts.reshape(-1, 3)
    return pts[:, :3]


def read_npy(path):
    pts = np.load(path).astype(np.float32)
    pts = np.squeeze(pts)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        pts = pts.reshape(-1, 3)
    return pts[:, :3]


def read_pcd(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PCD header: {path}")
            s = line.decode("utf-8", errors="ignore").strip()
            header.append(s)
            if s.startswith("DATA"):
                data_type = s.split()[1]
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

            if vals.size % 3 == 0:
                return vals.reshape(-1, 3).astype(np.float32)

            raise ValueError(f"Cannot parse ascii PCD: {path}, values={vals.size}")

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

        raise ValueError(f"Unsupported PCD type: {data_type}")



def force_xyz(pts, label="cloud"):
    pts = np.asarray(pts, dtype=np.float32)
    pts = np.squeeze(pts)

    if pts.ndim == 1:
        if pts.size % 3 != 0:
            raise ValueError(f"{label}: cannot reshape {pts.shape} to xyz")
        return pts.reshape(-1, 3)

    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts

    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        return pts.reshape(-1, 3)

    if pts.size % 3 == 0:
        return pts.reshape(-1, 3)

    raise ValueError(f"{label}: bad point cloud shape {pts.shape}")

def downsample(pts, n=2500):
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def set_equal_axes(ax, clouds):
    all_pts = np.vstack(clouds)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = (mn + mx) / 2.0
    radius = (mx - mn).max() / 2.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def plot_sample(row, tag):
    name = row["name"]

    partial = force_xyz(read_xyz(PARTIAL_DIR / f"{name}.xyz"), "partial")
    gt = force_xyz(read_xyz(GT_DIR / f"{name}.xyz"), "gt")
    g3d = force_xyz(read_pcd(G3D_DIR / name / "complete_pc.pcd"), "3DSGrasp")
    ada = force_xyz(read_npy(ADA_DIR / name / "fine.npy"), "AdaPoinTr")

    center = partial.mean(axis=0)
    scale = np.linalg.norm(partial - center[None, :], axis=1).max()

    partial_n = (partial - center[None, :]) / scale
    gt_n = (gt - center[None, :]) / scale
    g3d_n = (g3d - center[None, :]) / scale
    ada_n = ada

    clouds = [
        ("Partial input", downsample(partial_n)),
        (f"3DSGrasp\nCD={row['3dsgrasp_cd_l2_x1000']:.1f}", downsample(g3d_n)),
        (f"AdaPoinTr\nCD={row['adapointr_cd_l2_x1000']:.1f}", downsample(ada_n)),
        ("GT model", downsample(gt_n)),
    ]

    all_clouds = [pts for _, pts in clouds]

    fig = plt.figure(figsize=(15, 4))

    for i, (title, pts) in enumerate(clouds, start=1):
        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
        ax.set_title(title)
        set_equal_axes(ax, all_clouds)
        ax.view_init(elev=20, azim=45)

    fig.suptitle(name)
    plt.tight_layout()

    out = OUT_DIR / f"{tag}_{name}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print("Wrote", out)


def main():
    df = pd.read_csv(COMPARE)
    df["delta_ada_minus_3d"] = df["adapointr_cd_l2_x1000"] - df["3dsgrasp_cd_l2_x1000"]

    examples = [
        ("3dsgrasp_much_better", df.sort_values("delta_ada_minus_3d", ascending=False).iloc[0]),
        ("adapointr_best_case", df.sort_values("adapointr_improvement_percent", ascending=False).iloc[0]),
        ("adapointr_worst_case", df.sort_values("adapointr_improvement_percent", ascending=True).iloc[0]),
    ]

    median_idx = (df["delta_ada_minus_3d"] - df["delta_ada_minus_3d"].median()).abs().idxmin()
    examples.append(("typical_case", df.loc[median_idx]))

    for tag, row in examples:
        plot_sample(row, tag)

    print("")
    print("Done. Open:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
