from pathlib import Path
import re
import numpy as np
from PIL import Image

ROOT = Path("~/bop_datasets/ycbv_3dsgrasp_eval_filtered").expanduser()
YCBV_TEST = Path("~/bop_datasets/ycbv/test").expanduser()
PARTIAL_DIR = ROOT / "partials"
OUT_DIR = ROOT / "vlm_object_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_name(name):
    m = re.match(r"scene(\d+)_im(\d+)_gt(\d+)_obj(\d+)", name)
    if not m:
        raise ValueError(name)
    return tuple(map(int, m.groups()))

for partial in sorted(PARTIAL_DIR.glob("*.xyz")):
    name = partial.stem
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
        continue

    pad = 35
    h, w = mask.shape
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, w)
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, h)

    crop = rgb[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]

    # Red transparent mask overlay
    crop[mask_crop] = (0.6 * crop[mask_crop] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

    out = OUT_DIR / f"{name}.png"
    Image.fromarray(crop).save(out)

print("Wrote crops to:", OUT_DIR)
print("Number of crops:", len(list(OUT_DIR.glob('*.png'))))
