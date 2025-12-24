import os
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from tqdm import tqdm
import cv2

def save_pointclouds_from_bag(bag_path: str):
    topic_name = '/rslidar_points'
    #topic_name = '/rslidar_points_2'
    output_dir = os.path.join('saved_clouds_npz', topic_name.strip('/'))
    os.makedirs(output_dir, exist_ok=True)

    png_dir = os.path.join("saved_clouds", topic_name.strip('/'))
    os.makedirs(png_dir, exist_ok=True)

    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"❌ Error: the path '{bag_path}' does not exist.")
        return

    print(f"📂 Saving npz clouds in: {output_dir}")
    print(f"📂 Saving png clouds in: {png_dir}")
    print(f"📦 Reading bag file: {bag_path}\n")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == topic_name]
        total_msgs = sum(1 for _ in reader.messages(connections=connections))

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == topic_name]
        if not connections:
            print(f"⚠️ Topic {topic_name} was not found in the bag file. Check parameter topic_name in the script")
            return

        count = 0

        with tqdm(total=total_msgs, desc="Processing clouds", unit="msg") as pbar:
            for conn, _, rawdata in reader.messages(connections=connections):
                # Deserialize the message
                msg = reader.deserialize(rawdata, conn.msgtype)

                # Convert to a ROS2 message
                ros_msg = PointCloud2()
                ros_header = Header()
                ros_header.stamp.sec = msg.header.stamp.sec
                ros_header.stamp.nanosec = msg.header.stamp.nanosec
                ros_header.frame_id = msg.header.frame_id
                ros_msg.header = ros_header
                ros_msg.height = msg.height
                ros_msg.width = msg.width
                ros_fields = []
                for f in msg.fields:
                    pf = PointField()
                    pf.name = f.name
                    pf.offset = f.offset
                    pf.datatype = f.datatype
                    pf.count = f.count
                    ros_fields.append(pf)
                ros_msg.fields = ros_fields
                ros_msg.is_bigendian = msg.is_bigendian
                ros_msg.point_step = msg.point_step
                ros_msg.row_step = msg.row_step
                ros_msg.data = bytes(msg.data)
                ros_msg.is_dense = msg.is_dense

                # Obtain timestamp from each message
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                timestamp_str = f"{sec:02d}.{nsec:09d}"

                try:
                    cloud = np.array(list(pc2.read_points(
                        ros_msg,
                        field_names=("x", "y", "z", "intensity"),
                        skip_nans=True
                    )))
                    points = np.vstack([cloud['x'], cloud['y'], cloud['z'], cloud['intensity']]).T.astype(np.float32)

                except Exception as e:
                    print(f"❌ Error while reading cloud at time {timestamp_str}: {e}")
                    continue

                save_path = os.path.join(output_dir, f"{timestamp_str}.npz")
                np.savez_compressed(save_path, points=points)

                # Save the BEV projection of the point cloud in PNG format
                xlim = (-30, 30)
                ylim = (-30, 30)
                x, y = points[:, 0], points[:, 1]
                in_view = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])
                x, y = x[in_view], y[in_view]

                img_w, img_h = 1024, 1024
                xi = ((x - xlim[0]) / (xlim[1] - xlim[0]) * (img_w - 1)).astype(np.int32)
                yi = ((y - ylim[0]) / (ylim[1] - ylim[0]) * (img_h - 1)).astype(np.int32)
                yi = img_h - yi - 1 

                # Create image
                color_img = np.zeros((img_h, img_w), dtype=np.uint8)
                xi = np.clip(xi, 0, img_w - 1)
                yi = np.clip(yi, 0, img_h - 1)
                color_img[yi, xi] = 255

                # Create destination folder
                png_path = os.path.join(png_dir, f"{timestamp_str}.png")
                cv2.imwrite(png_path, color_img)
                count += 1
                pbar.update(1)

    print(f"\n Process completed. Number of saved clouds: {count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract point clouds from a ROS2 bag and saves it in npz format.")
    parser.add_argument("bag_path", help="Path to the .db3 file or bag folder")
    args = parser.parse_args()

    save_pointclouds_from_bag(args.bag_path)
