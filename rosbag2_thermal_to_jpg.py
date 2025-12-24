import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from sensor_msgs.msg import CompressedImage, Image

def save_thermal_images_from_bag(bag_path: str):
    topics = [
        '/thermal_cameras/camera1/image_compressed',
        '/thermal_cameras/camera2/image_compressed',
        '/thermal_cameras/camera3/image_compressed',
        '/thermal_cameras/camera4/image_compressed',
    ]

    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"❌ Error: path '{bag_path}' does not exist.")
        return

    print(f"📦 Reading bag: {bag_path}\n")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Calculate the total amount of images for the progress bar
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        total_msgs = sum(1 for _ in reader.messages(connections=connections))

    # Main loop: process and save images
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        if not connections:
            print("⚠️ Thermal camera topics not found in the bag.")
            return

        with tqdm(total=total_msgs, desc="Processing thermal images", unit="img") as pbar:
            for conn, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)
                topic_name = conn.topic

                # Timestamp
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                timestamp_str = f"{sec + nsec / 1e9:.9f}"

                # Define destination folder
                subdir = topic_name.strip('/').replace('/', '_')
                output_dir = Path("saved_images") / subdir
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = output_dir / f"image_{timestamp_str}.jpg"

                try:
                    msgtype = conn.msgtype

                    if "CompressedImage" in msgtype:
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

                    elif "Image" in msgtype:
                        if getattr(msg, "encoding", "") == "32FC1":
                            img_data = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
                            norm = cv2.normalize(img_data, None, 0, 255, cv2.NORM_MINMAX)
                            cv_image = norm.astype(np.uint8)
                        else:
                            cv_image = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)

                    else:
                        raise ValueError(f"Message type not recognized: {msgtype}")

                    # Save image in JPEG format with 95% compression
                    cv2.imwrite(str(filename), cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                except Exception as e:
                    print(f"❌ Error saving image from {topic_name}: {e}")
                pbar.update(1)

    print("\n Process completed. Images saved in saved_images/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract thermal images from a ROS2 bag and saves it as JPG.")
    parser.add_argument("bag_path", help="Path to the .db3 file or bag folder")
    args = parser.parse_args()

    save_thermal_images_from_bag(args.bag_path)
