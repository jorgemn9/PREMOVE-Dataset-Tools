import shutil
import yaml
from pathlib import Path


# Root directory of the project
SOURCE_ROOT = Path(".")

# Synchronization index containing sensor file paths and timestamps
SYNC_INDEX = SOURCE_ROOT / "sync_dataset/sync_index.yaml"

# Destination root for the generated dataset
DEST_ROOT = SOURCE_ROOT / "night_ampliacion_dataset/simbev/sweeps"

# Saved global sensor data
SAVED_GNSS = SOURCE_ROOT / "saved_gnss/gnss_data.yaml"
SAVED_IMU = SOURCE_ROOT / "saved_imu"

# Mapping from sensor topic names to destination folder names (cameras)
CAMERA_MAP = {
    "zed_multi_stereo_1_left_color_rect_image": "STEREO_1_LEFT",
    "zed_multi_stereo_2_left_color_rect_image": "STEREO_2_LEFT",
    "zed_multi_stereo_3_left_color_rect_image": "STEREO_3_LEFT",
    "zed_multi_stereo_1_right_color_rect_image": "STEREO_1_RIGHT",
    "zed_multi_stereo_2_right_color_rect_image": "STEREO_2_RIGHT",
    "zed_multi_stereo_3_right_color_rect_image": "STEREO_3_RIGHT",
    "thermal_cameras_camera1_image_compressed": "THERMAL_1",
    "thermal_cameras_camera2_image_compressed": "THERMAL_2",
    "thermal_cameras_camera3_image_compressed": "THERMAL_3",
    "thermal_cameras_camera4_image_compressed": "THERMAL_4",
}

# Mapping from LiDAR topics to destination folder names
LIDAR_MAP = {
    "rslidar_points": "LiDAR_1",
    "rslidar_points_2": "LiDAR_2",
}

# Mapping of IMU source filenames to standardized destination names
IMU_MAP = {
    "stereo1_imu.yaml": "IMU_1.yaml",
    "stereo2_imu.yaml": "IMU_2.yaml",
    "stereo3_imu.yaml": "IMU_3.yaml",
}


def copy_sensor(sensor_name, sensor_info, ref_ts):
    """
    Copy a sensor file (camera or LiDAR) to the dataset structure,
    renaming it using the reference timestamp.

    Parameters
    ----------
    sensor_name : str
        Sensor topic name
    sensor_info : dict
        Dictionary containing the source file path
    ref_ts : float or int
        Reference timestamp (used for naming)
    """
    src_path = Path(sensor_info["path"])
    dst_files = []

    # ---------- CAMERA ----------
    if sensor_name in CAMERA_MAP:
        if not src_path.exists():
            print(f"File not found: {src_path}")
            return

        folder = CAMERA_MAP[sensor_name]
        dst_dir = DEST_ROOT / folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Preserve file extension and rename using reference timestamp
        dst_file = dst_dir / f"{folder}-{ref_ts}{src_path.suffix.lower()}"
        dst_files.append((src_path, dst_file))

    # ---------- LIDAR ----------
    elif sensor_name in LIDAR_MAP:
        folder = LIDAR_MAP[sensor_name]
        dst_dir = DEST_ROOT / folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        # LiDAR data is expected as PNG + NPZ
        png_path = src_path
        npz_path = png_path.with_suffix(".npz")

        if not npz_path.exists():
            print(f"NPZ file not found: {npz_path}")
            return

        dst_npz = dst_dir / f"{folder}-{ref_ts}.npz"
        dst_jpg = dst_dir / f"{folder}-{ref_ts}.jpg"

        dst_files.append((npz_path, dst_npz))
        if png_path.exists():
            dst_files.append((png_path, dst_jpg))

    # Unsupported sensor
    else:
        return

    # ---------- COPY FILES ----------
    for src, dst in dst_files:
        shutil.copy2(src, dst)


def copy_gnss():
    """
    Copy the global GNSS file into the dataset.
    """
    if not SAVED_GNSS.exists():
        print("GNSS file not found")
        return

    dst_dir = DEST_ROOT / "GNSS"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SAVED_GNSS, dst_dir / "GNSS.yaml")


def copy_imu():
    """
    Copy IMU files into the dataset using standardized filenames.
    """
    dst_dir = DEST_ROOT / "IMU"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in IMU_MAP.items():
        src_file = SAVED_IMU / src_name
        if not src_file.exists():
            print(f"IMU file not found: {src_name}")
            continue
        shutil.copy2(src_file, dst_dir / dst_name)


def main():
    """
    Main dataset generation routine.
    Reads the synchronization index and copies all associated sensor data.
    """
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Load synchronization index
    with open(SYNC_INDEX, "r") as f:
        sync_data = yaml.safe_load(f)

    # Iterate over synchronized sets
    for set_entry in sync_data["sets"]:
        ref_ts = set_entry["ref_timestamp_s"]

        for sensor_name, sensor_info in set_entry["files"].items():
            copy_sensor(sensor_name, sensor_info, ref_ts)

    # Copy static sensors
    copy_gnss()
    copy_imu()

    print("Dataset creation completed.")


if __name__ == "__main__":
    main()
