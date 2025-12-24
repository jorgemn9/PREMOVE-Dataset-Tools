import yaml
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def find_imu_connections(reader):
    """Return all connections that publish sensor_msgs/msg/Imu"""
    return [
        c for c in reader.connections
        if c.msgtype == "sensor_msgs/msg/Imu"
    ]


def save_imu_data_from_bag(bag_path: Path, output_dir: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        imu_conns = find_imu_connections(reader)

        if not imu_conns:
            print(f"⚠️  No IMU topic found in {bag_path.name}")
            return

        if len(imu_conns) > 1:
            print(
                f"⚠️  Multiple IMU topics in {bag_path.name}, using first: "
                f"{imu_conns[0].topic}"
            )

        imu_conn = imu_conns[0]

        total_msgs = sum(
            1 for _ in reader.messages(connections=[imu_conn])
        )

    imu_data = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        imu_conn = find_imu_connections(reader)[0]

        with tqdm(
            total=total_msgs,
            desc=f"{bag_path.name} ({imu_conn.topic})",
            unit="msg"
        ) as pbar:
            for conn, _, rawdata in reader.messages(connections=[imu_conn]):
                msg = reader.deserialize(rawdata, conn.msgtype)

                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                timestamp = sec + nsec * 1e-9

                imu_data.append({
                    'timestamp': timestamp,
                    'orientation': {
                        'x': msg.orientation.x,
                        'y': msg.orientation.y,
                        'z': msg.orientation.z,
                        'w': msg.orientation.w,
                    },
                    'angular_velocity': {
                        'x': msg.angular_velocity.x,
                        'y': msg.angular_velocity.y,
                        'z': msg.angular_velocity.z,
                    },
                    'linear_acceleration': {
                        'x': msg.linear_acceleration.x,
                        'y': msg.linear_acceleration.y,
                        'z': msg.linear_acceleration.z,
                    },
                })
                pbar.update(1)

    output_file = output_dir / f"{bag_path.name}.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(imu_data, f, sort_keys=False)

    print(f"✅ Saved {output_file}")


def process_imu_bags_folder(imu_bags_dir: str):
    imu_bags_dir = Path(imu_bags_dir)

    output_dir = Path("saved_imu")
    output_dir.mkdir(exist_ok=True)

    bag_dirs = sorted(
        d for d in imu_bags_dir.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    )

    print(f"📂 Found {len(bag_dirs)} bags\n")

    for bag_dir in bag_dirs:
        print(f"📦 Processing {bag_dir.name}")
        save_imu_data_from_bag(bag_dir, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract IMU data from multiple ROS2 bags (auto topic detection)"
    )
    parser.add_argument(
        "imu_bags_dir",
        help="Path to IMU_Bags directory"
    )

    args = parser.parse_args()
    process_imu_bags_folder(args.imu_bags_dir)
