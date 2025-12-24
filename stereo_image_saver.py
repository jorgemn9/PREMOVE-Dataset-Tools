import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
import cv2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os

from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class DataSaver(Node):
    def __init__(self):
        super().__init__('data_saver')
        self.bridge = CvBridge()

        # Left-channel stereo rgb topics
        self.image_topics = [
            # '/zed_multi/stereo_1/rgb/color/rect/image',
            # '/zed_multi/stereo_2/rgb/color/rect/image',
            # '/zed_multi/stereo_3/rgb/color/rect/image'
            '/zed_multi/stereo_1/left/color/rect/image',
            '/zed_multi/stereo_1/right/color/rect/image',
            '/zed_multi/stereo_2/left/color/rect/image',
            '/zed_multi/stereo_2/right/color/rect/image',
            '/zed_multi/stereo_3/left/color/rect/image',
            '/zed_multi/stereo_3/right/color/rect/image',
        ]

        self.image_counts = {topic: 0 for topic in self.image_topics}
        self.subscribers = []

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        for topic in self.image_topics:
            sub = self.create_subscription(Image, topic, self.make_image_callback(topic), qos_profile)
            self.subscribers.append(sub)
            os.makedirs(f'saved_images/{self.topic_to_dir(topic)}', exist_ok=True)

    def topic_to_dir(self, topic_name):
        return topic_name.strip('/').replace('/', '_')

    def make_image_callback(self, topic_name):
        def callback(msg):
            try:
                encoding = msg.encoding
                subdir = self.topic_to_dir(topic_name)

                # Obtain message timestamp
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                timestamp_str = f"{timestamp:.9f}"

                # Define output filename
                filename = f"saved_images/{subdir}/image_{timestamp_str}.jpg"

                # Progress checker
                self.get_logger().info(
                    f"[{topic_name}] Image info — width: {msg.width}, height: {msg.height}, "
                    f"step: {msg.step}, is_bigendian: {msg.is_bigendian}"
                )

                # Save left-channel rgb image
                cv_image = self.bridge.imgmsg_to_cv2(msg)
                cv2.imwrite(filename, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                self.get_logger().info(f"Saved {topic_name}: {filename}")

            except Exception as e:
                self.get_logger().error(f"Error saving image from {topic_name}: {e}")
        return callback

def main(args=None):
    rclpy.init(args=args)
    node = DataSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
