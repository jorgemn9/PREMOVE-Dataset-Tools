import argparse
import json
import os
import re
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import yaml
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d

# Define the names each sensor folder
SENSOR_DIR_NAMES = [
    "rslidar_points",
    "rslidar_points_2",
    "thermal_cameras_camera1_image_compressed",
    "thermal_cameras_camera2_image_compressed",
    "thermal_cameras_camera3_image_compressed",
    "thermal_cameras_camera4_image_compressed",
    "zed_multi_stereo_1_left_color_rect_image",
    "zed_multi_stereo_2_left_color_rect_image",
    "zed_multi_stereo_3_left_color_rect_image",
    "zed_multi_stereo_1_right_color_rect_image",
    "zed_multi_stereo_2_right_color_rect_image",
    "zed_multi_stereo_3_right_color_rect_image",
]

# Define the sensors included in the collage
COLLAGE_REQUIRED = [
    "rslidar_points",
    "zed_multi_stereo_1_left_color_rect_image",
    "zed_multi_stereo_2_left_color_rect_image",
    "zed_multi_stereo_3_left_color_rect_image",
    "thermal_cameras_camera1_image_compressed",
    "thermal_cameras_camera2_image_compressed",
    "thermal_cameras_camera3_image_compressed",
    "thermal_cameras_camera4_image_compressed",
]

# Define the offsets for the collage visualization
Y_OFFSETS = {
    "zed_multi_stereo_1_left_color_rect_image": 200,  
    "zed_multi_stereo_3_left_color_rect_image": 200,   
    "thermal_cameras_camera1_image_compressed": -200,              
    "thermal_cameras_camera4_image_compressed": -200,           
}

IMAGE_PREFIX = "image_"
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

def find_sensor_dirs(root: Path) -> Dict[str, Path]:
    """
    Traverse the root and pick the best-matching directory for each sensor name.
    """
    found: Dict[str, List[Path]] = {name: [] for name in SENSOR_DIR_NAMES}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if base in found:
            found[base].append(Path(dirpath))
    chosen: Dict[str, Path] = {}
    for name, candidates in found.items():
        if not candidates:
            continue
        best = max(candidates, key=lambda p: sum(1 for _ in p.glob("*")))
        chosen[name] = best
    return chosen

def normalize_epoch_seconds(x: float) -> float:
    """
    Normalize timestamps to seconds (handles ms, us, ns).
    """
    if x > 1e14:      # ns
        return x / 1e9
    if x > 1e10:      # ms
        return x / 1e3
    return x          # seconds

def extract_timestamp_from_name(name: str, is_image: bool) -> Optional[float]:
    """ 
    Extract the first numeric token from the filename. 
    """
    stem = Path(name).stem
    if is_image and IMAGE_PREFIX in stem:
        after = stem.split(IMAGE_PREFIX, 1)[1]
        m = NUM_RE.search(after)
        if not m:
            return None
        raw = m.group(1)
    else:
        m = NUM_RE.search(stem)
        if not m:
            return None
        raw = m.group(1)
    try:
        val = float(raw)
    except ValueError:
        return None
    return normalize_epoch_seconds(val)

def list_files_with_ts(folder: Path, is_image: bool) -> List[Tuple[float, Path]]:
    """ 
    List files in a folder together with their extracted timestamps, sorted by time. 
    """
    items: List[Tuple[float, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        ts = extract_timestamp_from_name(p.name, is_image=is_image)
        if ts is None:
            continue
        items.append((ts, p))
    items.sort(key=lambda x: x[0])
    return items

def closest_within(sorted_list: List[Tuple[float, Path]], ts: float, tol_s: float) -> Optional[Tuple[float, Path, float]]:
    """
    Find the closest timestamped file to ts within tol_s seconds.
    Uses binary search on the sorted list.
    """
    if not sorted_list:
        return None
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    candidates = []
    for idx in (lo-1, lo, lo+1):
        if 0 <= idx < len(sorted_list):
            t2, p2 = sorted_list[idx]
            delta = abs(t2 - ts)
            if delta <= tol_s:
                candidates.append((t2, p2, delta))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[2])

def ensure_dir(path: Path):
    """ 
    Create directory if it does not exist. 
    """
    path.mkdir(parents=True, exist_ok=True)

def copy_like(src: Path, dst: Path, mode: str):
    """
    Copy or link files into the output dataset.
    Supports direct copy, hardlink and symlink.
    Falls back to copy on unsupported link types.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hard":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("Invalid --link mode (use: copy | hard | symlink)")

def write_indexes(out_dir: Path, sets: List[dict], formats: List[str]):
    """
    Write synchronization indexes in one or more formats (JSON, YAML, CSV).
    """
    # JSON
    if "json" in formats:
        with open(out_dir / "sync_index.json", "w") as f:
            json.dump({"sets": sets}, f, indent=2)
        print(f"Index written: {out_dir / 'sync_index.json'}")

    # YAML
    if "yaml" in formats:
        with open(out_dir / "sync_index.yaml", "w") as f:
            yaml.safe_dump({"sets": sets}, f, sort_keys=False)
        print(f"Index written: {out_dir / 'sync_index.yaml'}")

    # CSV
    if "csv" in formats:
        fieldnames = ["set", "ref_sensor", "ref_timestamp_s"]
        for s in SENSOR_DIR_NAMES:
            fieldnames += [f"{s}__timestamp_s", f"{s}__delta_ms_vs_ref", f"{s}__path"]
        with open(out_dir / "sync_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            # One row per synchronized set
            for entry in sets:
                row = {
                    "set": entry["set"],
                    "ref_sensor": entry["ref_sensor"],
                    "ref_timestamp_s": f"{entry['ref_timestamp_s']:.9f}",
                }

                # Fill per-sensor data (or leave empty if missing)
                for s in SENSOR_DIR_NAMES:
                    if s in entry["files"]:
                        info = entry["files"][s]
                        row[f"{s}__timestamp_s"] = f"{info['timestamp_s']:.9f}"
                        row[f"{s}__delta_ms_vs_ref"] = f"{info['delta_ms_vs_ref']:.3f}"
                        row[f"{s}__path"] = info["path"]
                    else:
                        row[f"{s}__timestamp_s"] = ""
                        row[f"{s}__delta_ms_vs_ref"] = ""
                        row[f"{s}__path"] = ""
                w.writerow(row)
        print(f"Index written: {out_dir / 'sync_index.csv'}")

# ---------- Collage helpers ----------
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

def _is_image_file(path: Path) -> bool:
    """Return True if the file extension corresponds to a supported image format."""
    return path.suffix.lower() in _IMAGE_EXTS

def _safe_load_image(path: Path, max_w: int, max_h: int, allow_upscale: bool = True) -> Optional["Image.Image"]:
    """
    Load an image and scale it to fit within (max_w, max_h) keeping aspect ratio.
    If allow_upscale=True, the image will be upscaled if it's smaller than the box.
    """
    if Image is None:
        return None
    try:
        img = Image.open(str(path)).convert("RGB")
        if allow_upscale:
            # Scales up or down to fit in the box, preserving aspect ratio
            img = ImageOps.contain(img, (max_w, max_h), Image.BICUBIC)
        else:
            img.thumbnail((max_w, max_h), Image.BICUBIC) # Downscale-only fallback
        return img
    except Exception:
        return None

def _load_point_cloud_points(pc_path: Path) -> Optional["np.ndarray"]:
    """
    Load point cloud data and return an Nx3 array (x, y, z).
    Tries NPZ first, then Open3D-supported formats, and finally
    falls back to ASCII PCD parser.
    """
    if np is None:
        return None

    if pc_path.suffix.lower() == ".npz":
        try:
            data = np.load(str(pc_path))
            for key in ("points", "xyz", "data"):
                if key in data:
                    pts = np.asarray(data[key], dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[1] >= 3:
                        return pts[:, :3]
            if len(data.files) > 0:
                first = data[data.files[0]]
                if first.ndim == 2 and first.shape[1] >= 3:
                    return first[:, :3].astype(np.float32)
        except Exception as e:
            print(f"⚠️ Error loading NPZ cloud {pc_path}: {e}")
            return None

    if o3d is not None and pc_path.suffix.lower() in (".pcd", ".ply"):
        try:
            pcd = o3d.io.read_point_cloud(str(pc_path))
            if pcd and len(pcd.points) > 0:
                return np.asarray(pcd.points, dtype=np.float32)
        except Exception:
            pass

    if pc_path.suffix.lower() == ".pcd":
        try:
            with open(pc_path, "r") as f:
                lines = f.readlines()
            data_start = 0
            fields = []
            for i, line in enumerate(lines):
                low = line.lower()
                if low.startswith("fields"):
                    fields = line.strip().split()[1:]
                if low.startswith("data"):
                    data_start = i + 1
                    break
            if not fields:
                fields = ["x", "y", "z"]
            idx_x = fields.index("x") if "x" in fields else 0
            idx_y = fields.index("y") if "y" in fields else 1
            idx_z = fields.index("z") if "z" in fields else 2
            pts = []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) <= max(idx_x, idx_y, idx_z):
                    continue
                try:
                    pts.append([float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])])
                except ValueError:
                    continue
            if pts:
                return np.array(pts, dtype=np.float32)
        except Exception:
            pass

    return None

def _render_point_cloud_to_image(pts: "np.ndarray", size: int = 800) -> Optional["Image.Image"]:
    """
    Render a point cloud as a top-down XY scatter image.
    Points are colored by normalized Z height.
    """
    if plt is None or Image is None or np is None:
        return None
    try:
        n = pts.shape[0]
        if n > 250_000:
            idx = np.random.choice(n, 250_000, replace=False)
            pts = pts[idx]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            c = (z - zmin) / (zmax - zmin)
        else:
            c = np.zeros_like(z)

        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, c=c, s=0.1, marker='.', linewidths=0)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        fig.tight_layout(pad=0)

        import io
        buf = io.BytesIO()
        fig.canvas.draw()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize((size, size), Image.BICUBIC)
        return img
    except Exception:
        return None

def _match_saved_cloud(root: Path, stem: str, prefer: str) -> Optional[Path]:
    """
    Find a saved cloud file by timestamp stem in:
      <root>/saved_clouds_npz/rslidar_points/{stem}.npz
      <root>/saved_clouds_pcd/rslidar_points/{stem}.pcd
    Respect 'prefer' when specified ('npz' or 'pcd'). For 'auto', try npz -> pcd.
    """
    npz_path = root / "saved_clouds_npz" / "rslidar_points" / f"{stem}.npz"
    pcd_path = root / "saved_clouds_pcd" / "rslidar_points" / f"{stem}.pcd"

    if prefer == "npz":
        return npz_path if npz_path.exists() else None
    if prefer == "pcd":
        return pcd_path if pcd_path.exists() else None

    if npz_path.exists():
        return npz_path
    if pcd_path.exists():
        return pcd_path
    return None

def create_collage_for_set(
    set_entry: dict,
    output_dir: Path,
    root: Path,
    lidar_source: str,
    center_size: int = 600,
    top_thumb_w: int = 720, top_thumb_h: int = 480,
    bot_thumb_w: int = 520, bot_thumb_h: int = 400,
    margin: int = 20, gap: int = 10,
) -> bool:
    """
    Build a collage image (saved as JPG):
      [       ZED1     ZED2     ZED3      ] Top row
      [             LiDAR_1               ] Central row
      [Thermal1 Thermal2 Thermal3 Thermal4] Bottom row
    Saves to output_dir/<set>.jpg
    Returns True on success, False if skipped/failed.
    """
    files = set_entry.get("files", {})

    # Ensure all required streams are present
    for name in COLLAGE_REQUIRED:
        if name not in files:
            return False
        if not Path(files[name]["path"]).exists():
            return False

    if Image is None:
        print("Pillow (PIL) not installed; skipping collages. Install with `pip install pillow`.")
        return False

    # ---- Center: LiDAR ----
    lidar_out_path = Path(files["rslidar_points"]["path"])
    stem = lidar_out_path.stem
    center_img = None

    lidar_png_path = lidar_out_path.parent / f"{stem}.png"
    if lidar_png_path.exists():
        img = _safe_load_image(lidar_png_path, center_size, center_size)
        if img is not None:
            center_img = img.resize((center_size, center_size), Image.BICUBIC)
    
    if center_img is None:
        if lidar_source in ("npz", "pcd", "auto"):
            src = _match_saved_cloud(root, stem, lidar_source)
            if src is not None:
                pts = _load_point_cloud_points(src)
                if pts is not None:
                    center_img = _render_point_cloud_to_image(pts, size=center_size)
    
    if center_img is None and _is_image_file(lidar_out_path):
        img = _safe_load_image(lidar_out_path, center_size, center_size)
        if img is not None:
            center_img = img.resize((center_size, center_size), Image.BICUBIC)

    if center_img is None:
        return False

    top_names = [
        "zed_multi_stereo_1_left_color_rect_image",
        "zed_multi_stereo_2_left_color_rect_image",
        "zed_multi_stereo_3_left_color_rect_image",
    ]
    top_imgs = []
    for n in top_names:
        p = Path(files[n]["path"])
        img = _safe_load_image(p, top_thumb_w, top_thumb_h)
        if img is None:
            img = Image.new("RGB", (top_thumb_w, top_thumb_h), (255, 255, 255))
        top_imgs.append(img)

    bot_names = [
        "thermal_cameras_camera1_image_compressed",
        "thermal_cameras_camera2_image_compressed",
        "thermal_cameras_camera3_image_compressed",
        "thermal_cameras_camera4_image_compressed",
    ]
    bot_imgs = []
    for n in bot_names:
        p = Path(files[n]["path"])
        img = _safe_load_image(p, bot_thumb_w, bot_thumb_h)
        # if img is None:
        #     img = Image.new("RGB", (bot_thumb_w, bot_thumb_h), (255, 255, 255))
        bot_imgs.append(img)

    top_row_w = sum(img.width for img in top_imgs) + gap * (len(top_imgs) - 1)
    bot_row_w = sum(img.width for img in bot_imgs) + gap * (len(bot_imgs) - 1)
    canvas_w = max(center_size, top_row_w, bot_row_w) + 2 * margin
    canvas_h = margin + top_thumb_h + gap + center_size + gap + bot_thumb_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))  # dark bg

    cx = (canvas_w - center_size) // 2
    cy = margin + top_thumb_h + gap
    canvas.paste(center_img, (cx, cy))

    x0 = (canvas_w - top_row_w) // 2
    y0 = margin
    x = x0
    for n, img in zip(top_names, top_imgs):
        offset = Y_OFFSETS.get(n, 0)
        y = y0 + (top_thumb_h - img.height) // 2 + offset
        canvas.paste(img, (x, y))
        x += img.width + gap

    x0 = (canvas_w - bot_row_w) // 2
    y0 = cy + center_size + gap
    x = x0
    for n, img in zip(bot_names, bot_imgs):
        offset = Y_OFFSETS.get(n, 0)
        y = y0 + (bot_thumb_h - img.height) // 2 + offset
        canvas.paste(img, (x, y))
        x += img.width + gap

    ensure_dir(output_dir)
    out_path = output_dir / f"{set_entry['set']}.jpg"
    try:
        canvas.save(out_path, format="JPEG", quality=95, optimize=True)
        return True
    except Exception as e:
        print(f"Warning: failed to save collage {out_path}: {e}")
        return False

def make_white_image(path: Path, size: Tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (255, 255, 255))
    img.save(path)

def make_black_image(path: Path, size: Tuple[int, int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, (0, 0, 0))
    img.save(path)

def make_empty_npz(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, points=np.zeros((0, 3), dtype=np.float32))

def main():
    parser = argparse.ArgumentParser(
        description="Synchronize sensor data by filename timestamps and (optionally) build collages."
    )
    parser.add_argument("root", type=str, help="Root folder containing the sensor subfolders.")
    parser.add_argument("--tolerance-ms", type=float, default=60.0,
                        help="Synchronization window (+/-) in milliseconds.")
    parser.add_argument("--out", type=str, default="sync_dataset",
                        help="Output folder (will be created if missing).")
    parser.add_argument("--link", type=str, default="copy",
                        choices=["copy", "hard", "symlink"],
                        help="How to materialize files in the output.")
    parser.add_argument("--ref-sensor", type=str, default="",
                        help="Force a specific reference sensor (optional).")
    parser.add_argument("--format", nargs="+", default=["json"],
                        choices=["json", "yaml", "csv"],
                        help="Index format(s) to write (one or more). Default: json")
    parser.add_argument("--make-collages", action="store_true",
                        help="If set, create collages for each synchronized set.")
    parser.add_argument("--collage-center-size", type=int, default=800,
                        help="Square size for the center LiDAR render (pixels).")
    parser.add_argument("--lidar-source", type=str, default="auto",
                        choices=["auto", "image", "npz", "pcd"],
                        help="Center LiDAR source: auto (NPZ->PCD->image), or force image/npz/pcd.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    tol_s = args.tolerance_ms / 1000.0

    sensor_dirs = find_sensor_dirs(root)
    non_empty: Dict[str, List[Tuple[float, Path]]] = {}
    for name, folder in sensor_dirs.items():
        is_image = name.startswith("thermal_cameras_") or name.startswith("zed_multi_stereo_")
        files = list_files_with_ts(folder, is_image=is_image)
        if files:
            non_empty[name] = files
        else:
            print(f"Warning: no valid files found in {name} ({folder})")

    if not non_empty:
        print("No valid files found. Aborting.")
        return

    if args.ref_sensor and args.ref_sensor in non_empty:
        ref_sensor = args.ref_sensor
    else:
        ref_sensor = max(non_empty.keys(), key=lambda k: len(non_empty[k]))
    ref_list = non_empty[ref_sensor]
    print(f"Reference sensor: {ref_sensor} ({len(ref_list)} files)")

    other_sensors = [s for s in SENSOR_DIR_NAMES if s in non_empty and s != ref_sensor]
    ensure_dir(out_dir)

    sets: List[dict] = []
    set_idx = 0

    written_paths = {}


    # Main synchronization loop
    for ref_ts, ref_path in ref_list:
        group = {ref_sensor: (ref_ts, ref_path, 0.0)}
        for s in other_sensors:
            match = closest_within(non_empty[s], ref_ts, tol_s)
            if match is not None:
                group[s] = match
            else:
                group[s] = (None, None, None)

        set_idx += 1
        set_dir = out_dir / f"set_{set_idx:06d}"
        for s, (ts, p, delta_abs) in group.items():
            if p is not None and p.exists():
                dst = set_dir / s / p.name
                copy_like(p, dst, args.link)
                written_paths[s] = dst
            else:
                sensor_dir = set_dir / s
                sensor_dir.mkdir(parents=True, exist_ok=True)

                # ---- Dummy data handling ----
                if s.startswith("zed_multi_stereo_"):
                    dummy_name = f"{s}_dummy.jpg"
                    dummy_path = sensor_dir / f"{s}_dummy.jpg"
                    make_white_image(sensor_dir / dummy_name, (1600, 900))
                    written_paths[s] = dummy_path

                elif s.startswith("thermal_cameras_"):
                    dummy_name = f"{s}_dummy.jpg"
                    dummy_path = sensor_dir / f"{s}_dummy.jpg"
                    make_white_image(sensor_dir / dummy_name, (320, 240))
                    written_paths[s] = dummy_path

                elif s in ("rslidar_points", "rslidar_points_2"):
                    stem = f"{s}_dummy"

                    png_path = sensor_dir / f"{stem}.png"
                    npz_path = sensor_dir / f"{stem}.npz"

                    # PNG dummy
                    make_black_image(sensor_dir / f"{stem}.png", (800, 800))

                    # NPZ nulo
                    make_empty_npz(sensor_dir / f"{stem}.npz")
                    written_paths[s] = png_path 
            if s in ("rslidar_points", "rslidar_points_2"):
                try:
                    stem = Path(p).stem

                    for subdir in ["saved_clouds_npz", "saved_clouds_pcd", "saved_clouds"]:
                        src_cloud_npz = root / subdir / s / f"{stem}.npz"
                        src_cloud_pcd = root / subdir / s / f"{stem}.pcd"
                        src_cloud_png = root / subdir / s / f"{stem}.png"

                        if src_cloud_npz.exists():
                            shutil.copy2(src_cloud_npz, dst.parent / src_cloud_npz.name)
                            print(f"✅ Copied NPZ for {s} (set {set_idx:06d}): {src_cloud_npz.name}")
                            break
                        elif src_cloud_pcd.exists():
                            shutil.copy2(src_cloud_pcd, dst.parent / src_cloud_pcd.name)
                            print(f"✅ Copied PCD for {s} (set {set_idx:06d}): {src_cloud_pcd.name}")
                            break
                        elif src_cloud_png.exists():
                            shutil.copy2(src_cloud_png, dst.parent / src_cloud_png.name)
                            print(f"🖼️ Copied PNG for {s} (set {set_idx:06d}): {src_cloud_png.name}")
                            break
                    else:
                        print(f"⚠️ No source cloud found for {s}/{stem}")


                except Exception as e:
                    print(f"❌ Error copying cloud for {s} (set {set_idx:06d}): {e}")

        set_entry = {
            "set": f"set_{set_idx:06d}",
            "ref_sensor": ref_sensor,
            "ref_timestamp_s": float(ref_ts),
            "files": {}
        }
        for s in SENSOR_DIR_NAMES:
            ts, p, delta_abs = group.get(s, (None, None, None))

            if s in written_paths:
                path_out = str(written_paths[s].as_posix())
            else:
                path_out = ""

            if ts is not None and p is not None:
                delta_ms = float((ts - ref_ts) * 1000.0)
                dummy = "False"
            else:
                delta_ms = ""
                dummy = "True"

            set_entry["files"][s] = {
                "timestamp_s": float(ts) if ts is not None else None,
                "delta_ms_vs_ref": delta_ms,
                "path": path_out,
                "dummy": dummy
            }
        sets.append(set_entry)

    if not sets:
        print("No groups met the tolerance across all sensors.")
        return

    write_indexes(out_dir, sets, args.format)
    print(f"Done. Created groups: {len(sets)}")
    print(f"Output dir: {out_dir}")

    if args.make_collages:
        from functools import partial
        from concurrent.futures import ProcessPoolExecutor, as_completed

        collages_dir = out_dir / "collages"
        builder = partial(
            create_collage_for_set,
            output_dir=collages_dir,
            root=root,
            lidar_source=args.lidar_source,
            center_size=args.collage_center_size
        )

        ok_count = skip_count = 0
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(builder, entry): entry for entry in sets}
            for fut in as_completed(futs):
                try:
                    if fut.result():
                        ok_count += 1
                    else:
                        skip_count += 1
                except Exception:
                    skip_count += 1

        print(f"Collages: created {ok_count}, skipped {skip_count}.")

if __name__ == "__main__":
    main()