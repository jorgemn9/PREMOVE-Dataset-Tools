import yaml
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# Load ublox_msgs ROS2 message
msg_dir = "/home/premove/ros2_ws/install/ublox_msgs/share/ublox_msgs/msg"
typestore = get_typestore(Stores.ROS2_HUMBLE)

for msg_file in Path(msg_dir).glob("*.msg"):
    with open(msg_file, "r") as f:
        msg_text = f.read()
        msg_types = get_types_from_msg(msg_text, f"ublox_msgs/msg/{msg_file.stem}")
        typestore.register(msg_types)



def decode_navpvt_mode(fix_type: int, flags: int) -> str:
    """
    Decode positioning mode from NAV-PVT fix_type and flags fields.
    """
    diffSoln = (flags >> 1) & 1          # bit 1
    carrSoln = (flags >> 6) & 0x3        # bits 6–7

    if fix_type < 3:
        return "NO_FIX"
    elif diffSoln == 0 and carrSoln == 0:
        return "AUTONOMOUS"
    elif diffSoln == 1 and carrSoln == 0:
        return "DGNSS"
    elif diffSoln == 1 and carrSoln == 1:
        return "RTK_FLOAT"
    elif diffSoln == 1 and carrSoln == 2:
        return "RTK_FIXED"
    else:
        return "UNKNOWN"

def extract_gps_data(bag_path: str):
    """
    Extract and synchronize GPS data from both receivers (left / right).
    - First, /fix and /navpvt messages are paired by index (1:1)
    - Then, left and right streams are synchronized by nearest timestamp
    """

    topics_fix = [
        "/PREMOVE/left/gps/fix",
        "/PREMOVE/right/gps/fix",
    ]
    topics_navpvt = [
        "/PREMOVE/left/gps/navpvt",
        "/PREMOVE/right/gps/navpvt",
    ]

    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"❌ Error: path '{bag_path}' does not exist.")
        return

    print(f"📦 Reading bag: {bag_path}\n")

    gps_fix = {"left": [], "right": []}
    gps_navpvt = {"left": [], "right": []}

    # Main loop: read bag file
    with AnyReader([bag_path], default_typestore=typestore) as reader:

        with tqdm(desc="Reading GNSS messages", unit="msg") as pbar:
            for conn, timestamp, rawdata in reader.messages():

                topic = conn.topic

                if topic not in topics_fix + topics_navpvt:
                    continue

                msg = reader.deserialize(rawdata, conn.msgtype)
                side = "left" if "left" in topic else "right"
                t_str = f"{timestamp // 1_000_000_000}.{timestamp % 1_000_000_000:09d}"

                if topic.endswith("/fix"):
                    gps_fix[side].append({
                        "timestamp": t_str,
                        "t_ns": timestamp,
                        "latitude": msg.latitude,
                        "longitude": msg.longitude,
                        "altitude": msg.altitude,
                    })

                elif topic.endswith("/navpvt"):
                    gps_navpvt[side].append({
                        "timestamp": t_str,
                        "t_ns": timestamp,
                        "gps_time": (
                            f"{msg.year:04d}-{msg.month:02d}-{msg.day:02d} "
                            f"{msg.hour:02d}:{msg.min:02d}:{msg.sec:02d}.{msg.nano:09d}"
                        ),
                        "num_sv": msg.num_sv,
                        "mode": decode_navpvt_mode(msg.fix_type, msg.flags),
                    })

                pbar.update(1)

    # Synchronizes FIX and NAVPVT messages
    def nearest(msgs, t):
        return min(msgs, key=lambda m: abs(m["t_ns"] - t))

    combined = {"left": [], "right": []}

    for side in ["left", "right"]:
        for fix in gps_fix[side]:
            if len(gps_navpvt[side]) == 0:
                continue
            nav = nearest(gps_navpvt[side], fix["t_ns"])
            combined[side].append({
                "timestamp": fix["timestamp"],
                "t_ns": fix["t_ns"],
                "latitude": fix["latitude"],
                "longitude": fix["longitude"],
                "altitude": fix["altitude"],
                "gps_time": nav["gps_time"],
                "num_sv": nav["num_sv"],
                "mode": nav["mode"],
            })

    # Synchronizes left and right GNSS data within a 60 ms tolerance window
    def nearest_combined(msgs, t):
        return min(msgs, key=lambda m: abs(m["t_ns"] - t))

    synchronized = []
    for l in combined["left"]:
        if len(combined["right"]) == 0:
            break
        r = nearest_combined(combined["right"], l["t_ns"])
        dt = abs(l["t_ns"] - r["t_ns"]) * 1e-9
        if dt < 0.06:  # 60 ms tolerance
            synchronized.append({
                "timestamp": l["timestamp"],
                "left": l,
                "right": r,
                "time_diff": dt,
            })

    print(f"🔗 Synced {len(synchronized)} pairs (tolerance < 60 ms)")

    # Save GNSS data
    output_dir = Path("saved_gnss")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gnss_data.yaml"

    for entry in synchronized:
        entry.pop("time_diff", None) 
        if "left" in entry:
            entry["left"].pop("t_ns", None)
        if "right" in entry:
            entry["right"].pop("t_ns", None)
        entry.pop("t_ns", None) 

    with open(output_file, "w") as f:
        yaml.dump(synchronized, f, sort_keys=False)

    print(f"\n Process completed. GNSS data saved to: {output_file}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and synchronizes GNSS FIX/NAVPVT messages from two GNSS receivers.")
    parser.add_argument("bag_path", help="Path to the .db3 file or the bag folder")
    args = parser.parse_args()

    extract_gps_data(args.bag_path)
