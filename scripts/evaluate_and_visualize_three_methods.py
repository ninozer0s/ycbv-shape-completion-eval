from pathlib import Path
import re
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image
import open3d as o3d

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
YCBV_TEST = Path("~/bop_datasets/ycbv/test").expanduser()

PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
G3D_DIR = ROOT / "completions"
ADA_DIR = ROOT / "adapointr_completions_norm"
SHAPE_DIR = ROOT / "shape_comp_completions"
MANIFEST = ROOT / "shape_comp_manifest.csv"

OUT_ANALYSIS = ROOT / "analysis_three_methods"
OUT_VIS = ROOT / "visualizations_three_methods_rgb"
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)


def force_xyz(pts):
    pts = np.asarray(pts, dtype=np.float32)
    pts = np.squeeze(pts)

    if pts.ndim == 1:
        if pts.size % 3 != 0:
            raise ValueError(f"Cannot reshape {pts.shape} to xyz")
        return pts.reshape(-1, 3)

    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts

    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        return pts.reshape(-1, 3)

    if pts.size % 3 == 0:
        return pts.reshape(-1, 3)

    raise ValueError(f"Bad point cloud shape {pts.shape}")


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


def read_3ds_cloud(path):
    if path.suffix == ".pcd":
        return read_pcd(path)
    if path.suffix == ".xyz":
        return read_xyz(path)
    raise ValueError(path)


def read_shape_comp_cloud(name, n_points=8192):
    folder = SHAPE_DIR / name
    files = list(folder.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No shape_comp output for {name}")

    path = files[0]

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        pts_m = np.asarray(pcd.points).astype(np.float32)
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        pts_m = np.asarray(pcd.points).astype(np.float32)

    # shape_comp uses meters; YCB-V partial/GT are in mm.
    return force_xyz(pts_m * 1000.0)


def chamfer_and_fscore(pred, gt, thresholds=(0.01, 0.02)):
    pred = force_xyz(pred)
    gt = force_xyz(gt)

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
        fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        key = str(th).replace(".", "p")
        out[f"precision_{key}"] = precision
        out[f"recall_{key}"] = recall
        out[f"fscore_{key}"] = fscore

    return out


def parse_name(name):
    m = re.match(r"scene(\d+)_im(\d+)_gt(\d+)_obj(\d+)", name)
    if not m:
        raise ValueError(name)
    return tuple(map(int, m.groups()))


def rgb_crop(name, pad=25):
    scene_id, im_id, gt_idx, obj_id = parse_name(name)
    scene = YCBV_TEST / f"{scene_id:06d}"

    rgb_path = scene / "rgb" / f"{im_id:06d}.png"
    if not rgb_path.exists():
        rgb_path = scene / "rgb" / f"{im_id:06d}.jpg"

    mask_path = scene / "mask_visib" / f"{im_id:06d}_{gt_idx:06d}.png"

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 0

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgb

    h, w = mask.shape
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, w)
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, h)

    crop = rgb[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]
    crop[mask_crop] = (0.55 * crop[mask_crop] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    return crop


def get_3ds_variants(name, center, scale):
    folder = G3D_DIR / name
    candidates = []

    files = [
        ("complete_pc", folder / "complete_pc.pcd"),
        ("complete_pc_x", folder / "complete_pc_x.pcd"),
        ("outputfile", folder / "outputfile.xyz"),
    ]

    for label, path in files:
        if not path.exists():
            continue

        pts = force_xyz(read_3ds_cloud(path))
        candidates.append((f"{label}_raw", pts))
        candidates.append((f"{label}_norm_by_partial", (pts - center[None, :]) / scale))

    if not candidates:
        raise FileNotFoundError(f"No 3DSGrasp output for {name}")

    return candidates


def downsample(pts, n=2500):
    pts = force_xyz(pts)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def set_axes(ax, clouds):
    pts = np.vstack([force_xyz(c) for c in clouds])
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2

    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def main():
    names = sorted([p.stem for p in PARTIAL_DIR.glob("*.xyz")])
    print("Samples:", len(names))

    # Choose correct 3DSGrasp coordinate variant globally.
    variant_scores = {}

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        gt_n = (gt - center[None, :]) / scale

        for variant_name, pts in get_3ds_variants(name, center, scale):
            cd = chamfer_and_fscore(pts, gt_n)["cd_l2_x1000"]
            variant_scores.setdefault(variant_name, []).append(cd)

    means = {k: float(np.mean(v)) for k, v in variant_scores.items()}
    best_3ds_variant = min(means, key=means.get)

    print("")
    print("3DSGrasp coordinate sanity check")
    print("=" * 60)
    for k, v in sorted(means.items()):
        print(f"{k:32s}: {v:.4f}")
    print("")
    print("Using 3DSGrasp variant:", best_3ds_variant)
    print("")

    rows = []

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")
        shape = read_shape_comp_cloud(name)

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale
        shape_n = (shape - center[None, :]) / scale

        g3d = dict(get_3ds_variants(name, center, scale))[best_3ds_variant]

        partial_m = chamfer_and_fscore(partial_n, gt_n)
        g3d_m = chamfer_and_fscore(g3d, gt_n)
        ada_m = chamfer_and_fscore(ada, gt_n)
        shape_m = chamfer_and_fscore(shape_n, gt_n)

        obj_id = int(name.split("_obj")[-1])

        cds = {
            "3DSGrasp": g3d_m["cd_l2_x1000"],
            "AdaPoinTr": ada_m["cd_l2_x1000"],
            "shape_comp": shape_m["cd_l2_x1000"],
        }

        row = {
            "name": name,
            "obj_id": obj_id,
            "partial_cd_l2_x1000": partial_m["cd_l2_x1000"],
            "3dsgrasp_cd_l2_x1000": g3d_m["cd_l2_x1000"],
            "adapointr_cd_l2_x1000": ada_m["cd_l2_x1000"],
            "shape_comp_cd_l2_x1000": shape_m["cd_l2_x1000"],
            "3dsgrasp_improvement_percent": (partial_m["cd_l2"] - g3d_m["cd_l2"]) / partial_m["cd_l2"] * 100,
            "adapointr_improvement_percent": (partial_m["cd_l2"] - ada_m["cd_l2"]) / partial_m["cd_l2"] * 100,
            "shape_comp_improvement_percent": (partial_m["cd_l2"] - shape_m["cd_l2"]) / partial_m["cd_l2"] * 100,
            "winner": min(cds, key=cds.get),
            "partial_fscore_0p02": partial_m["fscore_0p02"],
            "3dsgrasp_fscore_0p02": g3d_m["fscore_0p02"],
            "adapointr_fscore_0p02": ada_m["fscore_0p02"],
            "shape_comp_fscore_0p02": shape_m["fscore_0p02"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ANALYSIS / "metrics_three_methods.csv", index=False)

    summary = []
    summary.append("Three-method shape completion comparison on YCB-V")
    summary.append("=" * 70)
    summary.append(f"Samples: {len(df)}")
    summary.append(f"3DSGrasp variant: {best_3ds_variant}")
    summary.append("")
    summary.append(f"Mean Partial CD-L2 x1000:    {df['partial_cd_l2_x1000'].mean():.4f}")
    summary.append(f"Mean 3DSGrasp CD-L2 x1000:   {df['3dsgrasp_cd_l2_x1000'].mean():.4f}")
    summary.append(f"Mean AdaPoinTr CD-L2 x1000:  {df['adapointr_cd_l2_x1000'].mean():.4f}")
    summary.append(f"Mean shape_comp CD-L2 x1000: {df['shape_comp_cd_l2_x1000'].mean():.4f}")
    summary.append("")
    summary.append(f"3DSGrasp improved:   {(df['3dsgrasp_improvement_percent'] > 0).sum()} / {len(df)}")
    summary.append(f"AdaPoinTr improved:  {(df['adapointr_improvement_percent'] > 0).sum()} / {len(df)}")
    summary.append(f"shape_comp improved: {(df['shape_comp_improvement_percent'] > 0).sum()} / {len(df)}")
    summary.append("")
    summary.append("Wins by CD-L2:")
    summary.append(df["winner"].value_counts().to_string())

    (OUT_ANALYSIS / "summary_three_methods.txt").write_text("\n".join(summary))
    print("\n".join(summary))

    # Plot mean CD.
    labels = ["Partial", "3DSGrasp", "AdaPoinTr", "shape_comp"]
    means_plot = [
        df["partial_cd_l2_x1000"].mean(),
        df["3dsgrasp_cd_l2_x1000"].mean(),
        df["adapointr_cd_l2_x1000"].mean(),
        df["shape_comp_cd_l2_x1000"].mean(),
    ]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, means_plot)
    plt.ylabel("CD-L2 x1000 lower is better")
    plt.title("Mean shape completion error")
    plt.tight_layout()
    plt.savefig(OUT_ANALYSIS / "mean_cd_l2_three_methods.png", dpi=200)
    plt.close()

    # Per-sample plot.
    sdf = df.sort_values("partial_cd_l2_x1000").reset_index(drop=True)
    plt.figure(figsize=(11, 4))
    plt.plot(sdf.index, sdf["partial_cd_l2_x1000"], label="Partial")
    plt.plot(sdf.index, sdf["3dsgrasp_cd_l2_x1000"], label="3DSGrasp")
    plt.plot(sdf.index, sdf["adapointr_cd_l2_x1000"], label="AdaPoinTr")
    plt.plot(sdf.index, sdf["shape_comp_cd_l2_x1000"], label="shape_comp")
    plt.ylabel("CD-L2 x1000 lower is better")
    plt.xlabel("Samples sorted by partial error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ANALYSIS / "per_sample_cd_l2_three_methods.png", dpi=200)
    plt.close()

    # Visual examples.
    df["best_margin_3ds"] = df[["adapointr_cd_l2_x1000", "shape_comp_cd_l2_x1000"]].min(axis=1) - df["3dsgrasp_cd_l2_x1000"]
    df["best_margin_shape"] = df[["3dsgrasp_cd_l2_x1000", "adapointr_cd_l2_x1000"]].min(axis=1) - df["shape_comp_cd_l2_x1000"]

    examples = [
        ("3dsgrasp_best_margin", df.sort_values("best_margin_3ds", ascending=False).iloc[0]),
        ("shape_comp_best_margin", df.sort_values("best_margin_shape", ascending=False).iloc[0]),
        ("adapointr_best_case", df.sort_values("adapointr_improvement_percent", ascending=False).iloc[0]),
    ]

    median_idx = (df["3dsgrasp_cd_l2_x1000"] - df["3dsgrasp_cd_l2_x1000"].median()).abs().idxmin()
    examples.append(("typical_case", df.loc[median_idx]))

    for tag, row in examples:
        name = row["name"]

        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")
        shape = read_shape_comp_cloud(name)

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale
        shape_n = (shape - center[None, :]) / scale
        g3d = dict(get_3ds_variants(name, center, scale))[best_3ds_variant]

        clouds = [
            ("Partial input", downsample(partial_n)),
            (f"3DSGrasp\nCD={row['3dsgrasp_cd_l2_x1000']:.1f}", downsample(g3d)),
            (f"AdaPoinTr\nCD={row['adapointr_cd_l2_x1000']:.1f}", downsample(ada)),
            (f"shape_comp\nCD={row['shape_comp_cd_l2_x1000']:.1f}", downsample(shape_n)),
            ("GT model", downsample(gt_n)),
        ]

        all_clouds = [pts for _, pts in clouds]

        fig = plt.figure(figsize=(21, 4))

        ax0 = fig.add_subplot(1, 6, 1)
        ax0.imshow(rgb_crop(name))
        ax0.set_title("RGB crop + mask")
        ax0.axis("off")

        for i, (title, pts) in enumerate(clouds, start=2):
            ax = fig.add_subplot(1, 6, i, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
            ax.set_title(title)
            set_axes(ax, all_clouds)
            ax.view_init(elev=20, azim=45)

        fig.suptitle(name)
        plt.tight_layout()

        out = OUT_VIS / f"{tag}_{name}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote", out)

    print("")
    print("Analysis:", OUT_ANALYSIS)
    print("Visualizations:", OUT_VIS)


if __name__ == "__main__":
    main()
