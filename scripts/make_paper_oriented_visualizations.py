from pathlib import Path
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree
import open3d as o3d

ROOT = Path(os.environ.get("EVAL_ROOT", "~/bop_datasets/ycbv_3dsgrasp_eval_filtered")).expanduser()
REPO = Path(os.environ.get("REPO_ROOT", ".")).resolve()

PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
G3D_DIR = ROOT / "completions"
ADA_DIR = ROOT / "adapointr_completions_norm"
SHAPE_DIR = ROOT / "shape_comp_completions"

CROP_DIRS = [
    REPO / "results" / "vlm_object_crops",
    ROOT / "vlm_object_crops",
]

OUT_DIR = REPO / "results" / "three_methods" / "visualizations_rgb_paper_oriented"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def force_xyz(pts):
    pts = np.asarray(pts, dtype=np.float32)
    pts = np.squeeze(pts)

    if pts.ndim == 1:
        return pts.reshape(-1, 3)

    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts

    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        return pts.reshape(-1, 3)

    return pts.reshape(-1, 3)


def read_xyz(path):
    return force_xyz(np.loadtxt(path).astype(np.float32))


def read_npy(path):
    return force_xyz(np.load(path).astype(np.float32))


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
            if fields and len(fields) > 0 and vals.size % len(fields) == 0:
                arr = vals.reshape(-1, len(fields))
                cols = {name: i for i, name in enumerate(fields)}
                if all(k in cols for k in ["x", "y", "z"]):
                    return arr[:, [cols["x"], cols["y"], cols["z"]]].astype(np.float32)
            return force_xyz(vals)

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


def chamfer_cd_l2_x1000(pred, gt):
    pred = force_xyz(pred)
    gt = force_xyz(gt)

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    d1, _ = tree_gt.query(pred, k=1)
    d2, _ = tree_pred.query(gt, k=1)

    return float(((d1 ** 2).mean() + (d2 ** 2).mean()) * 1000.0)


def get_3ds_variants(name, center, scale):
    folder = G3D_DIR / name
    candidates = []

    paths = [
        ("complete_pc_raw", folder / "complete_pc.pcd", False),
        ("complete_pc_norm", folder / "complete_pc.pcd", True),
        ("complete_pc_x_raw", folder / "complete_pc_x.pcd", False),
        ("complete_pc_x_norm", folder / "complete_pc_x.pcd", True),
        ("outputfile_raw", folder / "outputfile.xyz", False),
        ("outputfile_norm", folder / "outputfile.xyz", True),
    ]

    for label, path, normalize in paths:
        if not path.exists():
            continue

        if path.suffix == ".pcd":
            pts = read_pcd(path)
        else:
            pts = read_xyz(path)

        pts = force_xyz(pts)

        if normalize:
            pts = (pts - center[None, :]) / scale

        candidates.append((label, pts))

    if not candidates:
        raise FileNotFoundError(f"No 3DSGrasp output for {name}")

    return candidates


def read_shape_comp(name):
    folder = SHAPE_DIR / name
    files = sorted(folder.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No shape_comp output for {name}")

    path = files[0]

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=8192)
        pts_m = np.asarray(pcd.points).astype(np.float32)
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        pts_m = np.asarray(pcd.points).astype(np.float32)

    return force_xyz(pts_m * 1000.0)


def find_crop(name):
    for d in CROP_DIRS:
        p = d / f"{name}.png"
        if p.exists():
            return p
    return None


def downsample(pts, n=5000):
    pts = force_xyz(pts)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def plot_camera_view(ax, pts, title, xlim, ylim):
    pts = downsample(pts)

    # Camera-oriented projection:
    # horizontal = camera X, vertical = -camera Y.
    # This makes the point cloud orientation match the RGB image.
    x = pts[:, 0]
    y = -pts[:, 1]
    z = pts[:, 2]

    ax.scatter(x, y, s=0.35, c=z, cmap="viridis")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def main():
    names = sorted([p.stem for p in PARTIAL_DIR.glob("*.xyz")])
    print("Samples:", len(names))

    print("Choosing best 3DSGrasp coordinate variant...")
    scores = {}

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        gt_n = (gt - center[None, :]) / scale

        for label, pts in get_3ds_variants(name, center, scale):
            cd = chamfer_cd_l2_x1000(pts, gt_n)
            scores.setdefault(label, []).append(cd)

    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best_variant = min(means, key=means.get)

    print("3DSGrasp variants:")
    for k, v in sorted(means.items()):
        print(f"{k}: {v:.4f}")
    print("Using:", best_variant)

    written = 0

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")
        shape = read_shape_comp(name)

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale
        shape_n = (shape - center[None, :]) / scale

        g3d = dict(get_3ds_variants(name, center, scale))[best_variant]

        clouds = [
            ("Partial", partial_n),
            ("3DSGrasp", g3d),
            ("AdaPoinTr", ada),
            ("shape_comp", shape_n),
            ("GT", gt_n),
        ]

        all_pts = np.vstack([force_xyz(c) for _, c in clouds])
        x_vals = all_pts[:, 0]
        y_vals = -all_pts[:, 1]

        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())

        pad = 0.08 * max(x_max - x_min, y_max - y_min)
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min - pad, y_max + pad)

        fig = plt.figure(figsize=(16, 3.2))

        ax0 = fig.add_subplot(1, 6, 1)
        crop_path = find_crop(name)
        if crop_path is not None:
            ax0.imshow(Image.open(crop_path).convert("RGB"))
            ax0.set_title("RGB crop + mask", fontsize=9)
        else:
            ax0.text(0.5, 0.5, "No crop", ha="center", va="center")
            ax0.set_title("RGB crop + mask", fontsize=9)
        ax0.axis("off")

        for i, (title, pts) in enumerate(clouds, start=2):
            ax = fig.add_subplot(1, 6, i)
            plot_camera_view(ax, pts, title, xlim, ylim)

        fig.suptitle(name, fontsize=9)
        plt.tight_layout()

        out = OUT_DIR / f"{name}_paper_oriented.png"
        plt.savefig(out, dpi=250)
        plt.close()

        written += 1
        print("Wrote", out)

    print("Done")
    print("Wrote visualizations:", written)
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
