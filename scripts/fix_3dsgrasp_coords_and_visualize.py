from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
YCBV_TEST = Path("~/bop_datasets/ycbv/test").expanduser()

PARTIAL_DIR = ROOT / "partials"
GT_DIR = ROOT / "gt"
G3D_DIR = ROOT / "completions"
ADA_DIR = ROOT / "adapointr_completions_norm"

OUT_DIR = ROOT / "visualizations_compare_rgb_FIXED"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_DIR = ROOT / "analysis_compare_FIXED"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def force_xyz(pts):
    pts = np.asarray(pts, dtype=np.float32)
    pts = np.squeeze(pts)

    if pts.ndim == 1:
        return pts.reshape(-1, 3)

    if pts.ndim == 2 and pts.shape[1] == 3:
        return pts

    if pts.ndim == 2 and pts.shape[0] == 1 and pts.shape[1] % 3 == 0:
        return pts.reshape(-1, 3)

    if pts.size % 3 == 0:
        return pts.reshape(-1, 3)

    raise ValueError(f"Bad point shape: {pts.shape}")


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
            vals = np.fromstring(f.read().decode("utf-8", errors="ignore"), sep=" ", dtype=np.float32)

            if fields and vals.size % len(fields) == 0:
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


def read_cloud(path):
    if path.suffix == ".pcd":
        return read_pcd(path)
    if path.suffix == ".xyz":
        return read_xyz(path)
    if path.suffix == ".npy":
        return read_npy(path)
    raise ValueError(path)


def chamfer(pred, gt):
    pred = force_xyz(pred)
    gt = force_xyz(gt)
    t_gt = cKDTree(gt)
    t_pred = cKDTree(pred)
    d1, _ = t_gt.query(pred, k=1)
    d2, _ = t_pred.query(gt, k=1)
    return float(((d1 ** 2).mean() + (d2 ** 2).mean()) * 1000.0)


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

        pts = force_xyz(read_cloud(path))

        candidates.append((f"{label}_raw", pts))
        candidates.append((f"{label}_norm_by_partial", (pts - center[None, :]) / scale))

    return candidates


def main():
    rows = []
    variant_scores = {}

    names = sorted([p.stem for p in PARTIAL_DIR.glob("*.xyz")])

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale

        for variant_name, g3d in get_3ds_variants(name, center, scale):
            cd = chamfer(g3d, gt_n)
            variant_scores.setdefault(variant_name, []).append(cd)

    print("")
    print("3DSGrasp coordinate sanity check")
    print("=" * 60)
    means = {}
    for k, vals in sorted(variant_scores.items()):
        means[k] = float(np.mean(vals))
        print(f"{k:32s} mean CD-L2 x1000 = {means[k]:.4f}")

    best_variant = min(means, key=means.get)
    print("")
    print("USING BEST 3DSGRASP VARIANT:", best_variant)
    print("")

    for name in names:
        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale

        variants = dict(get_3ds_variants(name, center, scale))
        g3d = variants[best_variant]

        partial_cd = chamfer(partial_n, gt_n)
        g3d_cd = chamfer(g3d, gt_n)
        ada_cd = chamfer(ada, gt_n)

        rows.append({
            "name": name,
            "obj_id": int(name.split("_obj")[-1]),
            "partial_cd_l2_x1000": partial_cd,
            "3dsgrasp_cd_l2_x1000": g3d_cd,
            "adapointr_cd_l2_x1000": ada_cd,
            "3dsgrasp_improvement_percent": (partial_cd - g3d_cd) / partial_cd * 100,
            "adapointr_improvement_percent": (partial_cd - ada_cd) / partial_cd * 100,
            "winner": "3DSGrasp" if g3d_cd < ada_cd else "AdaPoinTr",
        })

    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS_DIR / "compare_fixed.csv", index=False)

    summary = []
    summary.append("FIXED comparison")
    summary.append("=" * 60)
    summary.append(f"Chosen 3DSGrasp variant: {best_variant}")
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

    (ANALYSIS_DIR / "summary_fixed.txt").write_text("\n".join(summary))
    print("\n".join(summary))

    df["delta_ada_minus_3d"] = df["adapointr_cd_l2_x1000"] - df["3dsgrasp_cd_l2_x1000"]

    examples = [
        ("3dsgrasp_much_better", df.sort_values("delta_ada_minus_3d", ascending=False).iloc[0]),
        ("adapointr_best_case", df.sort_values("adapointr_improvement_percent", ascending=False).iloc[0]),
        ("adapointr_worst_case", df.sort_values("adapointr_improvement_percent", ascending=True).iloc[0]),
    ]

    median_idx = (df["delta_ada_minus_3d"] - df["delta_ada_minus_3d"].median()).abs().idxmin()
    examples.append(("typical_case", df.loc[median_idx]))

    for tag, row in examples:
        name = row["name"]

        partial = read_xyz(PARTIAL_DIR / f"{name}.xyz")
        gt = read_xyz(GT_DIR / f"{name}.xyz")
        ada = read_npy(ADA_DIR / name / "fine.npy")

        center = partial.mean(axis=0)
        scale = np.linalg.norm(partial - center[None, :], axis=1).max()

        partial_n = (partial - center[None, :]) / scale
        gt_n = (gt - center[None, :]) / scale
        g3d = dict(get_3ds_variants(name, center, scale))[best_variant]

        clouds = [
            ("Partial input", downsample(partial_n)),
            (f"3DSGrasp\nCD={row['3dsgrasp_cd_l2_x1000']:.1f}", downsample(g3d)),
            (f"AdaPoinTr\nCD={row['adapointr_cd_l2_x1000']:.1f}", downsample(ada)),
            ("GT model", downsample(gt_n)),
        ]

        all_clouds = [pts for _, pts in clouds]

        fig = plt.figure(figsize=(18, 4))

        ax0 = fig.add_subplot(1, 5, 1)
        ax0.imshow(rgb_crop(name))
        ax0.set_title("RGB crop + mask")
        ax0.axis("off")

        for i, (title, pts) in enumerate(clouds, start=2):
            ax = fig.add_subplot(1, 5, i, projection="3d")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
            ax.set_title(title)
            set_axes(ax, all_clouds)
            ax.view_init(elev=20, azim=45)

        fig.suptitle(name)
        plt.tight_layout()

        out = OUT_DIR / f"{tag}_{name}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote", out)

    print("")
    print("Open:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
