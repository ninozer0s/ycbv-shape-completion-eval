from pathlib import Path
import csv
import numpy as np
import open3d as o3d

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
PARTIAL_DIR = ROOT / "partials"
OUT_DIR = ROOT / "shape_comp_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# shape_comp checkpoint classes:
# apple, bottle, bowl, box, can, hammer
#
# YCB-V object IDs:
# 1 master_chef_can, 2 cracker_box, 3 sugar_box, 4 tomato_soup_can,
# 5 mustard_bottle, 6 tuna_fish_can, 7 pudding_box, 8 gelatin_box,
# 9 potted_meat_can, 10 banana, 11 pitcher_base, 12 bleach_cleanser,
# 13 bowl, 14 mug, 15 power_drill, 16 wood_block, 17 scissors,
# 18 large_marker, 19 large_clamp, 20 extra_large_clamp, 21 foam_brick
OBJ_TO_CLASS = {
    1: "can",
    2: "box",
    3: "box",
    4: "can",
    5: "bottle",
    6: "can",
    7: "box",
    8: "box",
    9: "can",
    10: "apple",    # banana approximated by apple
    11: "bottle",
    12: "bottle",
    13: "bowl",
    14: "bowl",     # mug approximated by bowl
    15: "hammer",   # drill approximated by tool-like hammer class
    16: "box",
    17: "hammer",   # scissors approximated by tool-like hammer class
    18: "bottle",   # marker approximated by elongated bottle class
    19: "hammer",   # clamp approximated by tool-like hammer class
    20: "hammer",   # clamp approximated by tool-like hammer class
    21: "box",
}

rows = []

for partial_path in sorted(PARTIAL_DIR.glob("*.xyz")):
    name = partial_path.stem
    obj_id = int(name.split("_obj")[-1])

    cls = OBJ_TO_CLASS[obj_id]

    pts_mm = np.loadtxt(partial_path).astype(np.float32)
    if pts_mm.ndim == 1:
        pts_mm = pts_mm.reshape(-1, 3)

    # YCB-V points are in mm; shape_comp example point clouds are metric scale.
    pts_m = pts_mm[:, :3] / 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_m)

    out_ply = OUT_DIR / f"{name}.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=True)

    rows.append({
        "name": name,
        "obj_id": obj_id,
        "shape_comp_class": cls,
        "input_ply": str(out_ply),
        "output_ply": str(ROOT / "shape_comp_completions" / name / f"{cls}_completed.ply"),
    })

manifest = ROOT / "shape_comp_manifest.csv"
with open(manifest, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["name", "obj_id", "shape_comp_class", "input_ply", "output_ply"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Prepared shape_comp inputs:", len(rows))
print("Wrote manifest:", manifest)
print("Output dir:", OUT_DIR)
