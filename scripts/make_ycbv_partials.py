#!/usr/bin/env python3
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


YCBV_ROOT = Path.home() / "bop_datasets" / "ycbv"
OUT_ROOT = Path.home() / "bop_datasets" / "ycbv_3dsgrasp_eval"

NUM_PARTIAL_POINTS = 2048
NUM_GT_POINTS = 8192
RANDOM_SEED = 42


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_xyz(path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points.astype(np.float32), fmt="%.6f")


def sample_points(points, n, seed=0):
    rng = np.random.default_rng(seed)
    points = np.asarray(points)
    if len(points) == 0:
        return points
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace)
    return points[idx]


def depth_mask_to_points(depth, mask, cam_K, depth_scale):
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]

    ys, xs = np.where((mask > 0) & (depth > 0))

    z = depth[ys, xs].astype(np.float32) * float(depth_scale)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def load_model_points(models_dir, obj_id, n_points, seed):
    model_path = models_dir / f"obj_{obj_id:06d}.ply"
    mesh = trimesh.load(model_path, process=False)

    if hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    else:
        pts = np.asarray(mesh.vertices)
        pts = sample_points(pts, n_points, seed)

    return np.asarray(pts, dtype=np.float32)


def transform_model_to_camera(model_points, gt_entry):
    R = np.array(gt_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    t = np.array(gt_entry["cam_t_m2c"], dtype=np.float32).reshape(1, 3)
    return model_points @ R.T + t


def choose_one_image_per_scene(test_dir, targets_path):
    scene_to_im = {}

    if targets_path.exists():
        targets = load_json(targets_path)
        for item in targets:
            scene_id = int(item["scene_id"])
            im_id = int(item["im_id"])
            if scene_id not in scene_to_im:
                scene_to_im[scene_id] = im_id
        return scene_to_im

    for scene_path in sorted(test_dir.iterdir()):
        if not scene_path.is_dir():
            continue
        scene_id = int(scene_path.name)
        depth_dir = scene_path / "depth"
        images = sorted(depth_dir.glob("*.png"))
        if images:
            scene_to_im[scene_id] = int(images[0].stem)

    return scene_to_im


def main():
    test_dir = YCBV_ROOT / "test"
    models_dir = YCBV_ROOT / "models"
    targets_path = YCBV_ROOT / "test_targets_bop19.json"

    partial_dir = OUT_ROOT / "partials"
    gt_dir = OUT_ROOT / "gt"
    meta_path = OUT_ROOT / "metadata.csv"

    partial_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    scene_to_im = choose_one_image_per_scene(test_dir, targets_path)

    print(f"YCBV root: {YCBV_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Selected scenes: {len(scene_to_im)}")

    rows = []

    for scene_id, im_id in tqdm(sorted(scene_to_im.items())):
        scene_path = test_dir / f"{scene_id:06d}"

        scene_camera_path = scene_path / "scene_camera.json"
        scene_gt_path = scene_path / "scene_gt.json"

        if not scene_camera_path.exists() or not scene_gt_path.exists():
            print(f"[WARN] Missing scene files in {scene_path}")
            continue

        scene_camera = load_json(scene_camera_path)
        scene_gt = load_json(scene_gt_path)

        im_key = str(im_id)

        if im_key not in scene_camera or im_key not in scene_gt:
            print(f"[WARN] Missing image {scene_id:06d}/{im_id:06d}")
            continue

        cam_info = scene_camera[im_key]
        cam_K = np.array(cam_info["cam_K"], dtype=np.float32).reshape(3, 3)
        depth_scale = cam_info.get("depth_scale", 1.0)

        depth_path = scene_path / "depth" / f"{im_id:06d}.png"
        if not depth_path.exists():
            print(f"[WARN] Missing depth: {depth_path}")
            continue

        depth = imageio.imread(depth_path)
        gt_entries = scene_gt[im_key]

        for gt_idx, gt_entry in enumerate(gt_entries):
            obj_id = int(gt_entry["obj_id"])

            mask_path = scene_path / "mask_visib" / f"{im_id:06d}_{gt_idx:06d}.png"
            if not mask_path.exists():
                print(f"[WARN] Missing mask: {mask_path}")
                continue

            mask = imageio.imread(mask_path)
            partial_points = depth_mask_to_points(depth, mask, cam_K, depth_scale)

            if len(partial_points) < 50:
                print(f"[WARN] Too few partial points: scene={scene_id}, im={im_id}, gt_idx={gt_idx}, n={len(partial_points)}")
                continue

            sample_name = f"scene{scene_id:06d}_im{im_id:06d}_gt{gt_idx:06d}_obj{obj_id:06d}"

            partial_points = sample_points(
                partial_points,
                NUM_PARTIAL_POINTS,
                seed=RANDOM_SEED + scene_id + im_id + gt_idx,
            )

            model_points = load_model_points(
                models_dir,
                obj_id,
                NUM_GT_POINTS,
                seed=RANDOM_SEED + obj_id,
            )

            gt_points_cam = transform_model_to_camera(model_points, gt_entry)

            partial_path = partial_dir / f"{sample_name}.xyz"
            gt_path = gt_dir / f"{sample_name}.xyz"

            save_xyz(partial_path, partial_points)
            save_xyz(gt_path, gt_points_cam)

            rows.append({
                "sample_name": sample_name,
                "scene_id": scene_id,
                "im_id": im_id,
                "gt_idx": gt_idx,
                "obj_id": obj_id,
                "partial_path": str(partial_path),
                "gt_path": str(gt_path),
                "num_partial_points": len(partial_points),
                "num_gt_points": len(gt_points_cam),
            })

    df = pd.DataFrame(rows)
    df.to_csv(meta_path, index=False)

    print()
    print(f"Saved object samples: {len(df)}")
    print(f"Partials: {partial_dir}")
    print(f"GT:       {gt_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
