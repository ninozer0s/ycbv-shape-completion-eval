from pathlib import Path
import numpy as np
import csv

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
PARTIAL_DIR = ROOT / "partials"
OUT_DIR = ROOT / "adapointr_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for p in sorted(PARTIAL_DIR.glob("*.xyz")):
    pts = np.loadtxt(p).astype(np.float32)

    center = pts.mean(axis=0)
    scale = np.linalg.norm(pts - center[None, :], axis=1).max()

    if scale <= 0:
        print("Skipping invalid:", p.name)
        continue

    pts_norm = (pts - center[None, :]) / scale

    out = OUT_DIR / p.name
    np.savetxt(out, pts_norm, fmt="%.8f")

    rows.append({
        "name": p.stem,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "scale": scale,
    })

with open(ROOT / "adapointr_transforms.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "center_x", "center_y", "center_z", "scale"])
    writer.writeheader()
    writer.writerows(rows)

print("Wrote", len(rows), "normalized AdaPoinTr inputs to", OUT_DIR)
print("Wrote transforms to", ROOT / "adapointr_transforms.csv")
