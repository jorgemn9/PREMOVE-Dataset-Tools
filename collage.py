from pathlib import Path
from PIL import Image, ImageOps
import re

ROOT = Path("dataset_censored") #Path to your dataset folder
OUT_DIR = ROOT.parent / "collages_censored"
OUT_DIR.mkdir(exist_ok=True)

# Define ZED folders
ZED_DIRS = [
    ROOT / "STEREO_1_LEFT",
    ROOT / "STEREO_2_LEFT",
    ROOT / "STEREO_3_LEFT",
]

# Define thermal images folders
THERMAL_DIRS = [
    ROOT / "THERMAL_1",
    ROOT / "THERMAL_2",
    ROOT / "THERMAL_3",
    ROOT / "THERMAL_4",
]

# Define bottom LiDAR folder
LIDAR_DIR = ROOT / "LiDAR_1"

Y_OFFSETS = {
    "STEREO_1_LEFT": 200,
    "STEREO_3_LEFT": 200,
    "THERMAL_1": -200,
    "THERMAL_4": -200,
}

# ---------- PARAMS ----------
TOL_S = 0.005  # 500 ms Synchronization between data if needed, not in this case
TOP_SIZE = (720, 480)
BOT_SIZE = (520, 400)
CENTER_SIZE = 600
MARGIN = 20
GAP = 10
# ---------------------------


def extract_timestamp(path: Path):
    """
    Extract timestamp from filenames
    """
    try:
        return float(path.stem.split("-")[-1])
    except Exception:
        return None


def list_images(folder: Path):
    """
    Make a list of available images from each folder
    """
    items = []
    for p in folder.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            ts = extract_timestamp(p)
            if ts is not None:
                items.append((ts, p))
    return sorted(items, key=lambda x: x[0])


def closest_image(items, ts_ref):
    """
    For synchronization purposes. Not needed in this case
    """
    best = None
    best_dt = float("inf")
    for ts, p in items:
        dt = abs(ts - ts_ref)
        if dt < best_dt and dt <= TOL_S:
            best = p
            best_dt = dt
    return best


def safe_load(path, size):
    """
    Save images to file
    """
    img = Image.open(path).convert("RGB")
    return ImageOps.contain(img, size)


# ---------- INDEX DATA ----------
lidar_data = list_images(LIDAR_DIR)
zed_data = [list_images(d) for d in ZED_DIRS]
thermal_data = [list_images(d) for d in THERMAL_DIRS]

print("LiDAR images:", len(lidar_data))
for i, z in enumerate(zed_data):
    print(f"ZED {i+1} images:", len(z))
for i, t in enumerate(thermal_data):
    print(f"THERMAL {i+1} images:", len(t))

if not lidar_data:
    print("❌ No LiDAR images found. Abort.")
    exit()


# ---------- BUILD COLLAGES ----------
count = 0

for ts_lidar, lidar_path in lidar_data:
    zed_imgs = []
    for items in zed_data:
        p = closest_image(items, ts_lidar)
        if p is None:
            break
        zed_imgs.append(safe_load(p, TOP_SIZE))
    else:
        thermal_imgs = []
        for items in thermal_data:
            p = closest_image(items, ts_lidar)
            if p is None:
                break
            thermal_imgs.append(safe_load(p, BOT_SIZE))
        else:
            center_img = safe_load(lidar_path, (CENTER_SIZE, CENTER_SIZE))
            center_img = center_img.resize((CENTER_SIZE, CENTER_SIZE))

            top_w = sum(i.width for i in zed_imgs) + GAP * 2
            bot_w = sum(i.width for i in thermal_imgs) + GAP * 3

            canvas_w = max(top_w, CENTER_SIZE, bot_w) + 2 * MARGIN
            canvas_h = (
                MARGIN +
                TOP_SIZE[1] +
                GAP +
                CENTER_SIZE +
                GAP +
                BOT_SIZE[1] +
                MARGIN
            )

            canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

            # Center
            cx = (canvas_w - CENTER_SIZE) // 2
            cy = MARGIN + TOP_SIZE[1] + GAP
            canvas.paste(center_img, (cx, cy))

            # Top (ZED)
            x = (canvas_w - top_w) // 2
            y = MARGIN
            for idx, img in enumerate(zed_imgs):
                sensor = ZED_DIRS[idx].name
                offset = Y_OFFSETS.get(sensor, 0)
                canvas.paste(img, (x, y + offset))
                x += img.width + GAP

            # Bottom (THERMAL)
            x = (canvas_w - bot_w) // 2
            y = cy + CENTER_SIZE + GAP
            for idx, img in enumerate(thermal_imgs):
                sensor = THERMAL_DIRS[idx].name
                offset = Y_OFFSETS.get(sensor, 0)
                canvas.paste(img, (x, y + offset))
                x += img.width + GAP

            out = OUT_DIR / f"collage-{ts_lidar:.6f}.jpg"
            canvas.save(out, quality=95)
            count += 1
            print("Collage:", out.name)

print(f"\n TOTAL number of collages created: {count}")
