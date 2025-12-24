import argparse
import rosbag2_py
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import rclpy.serialization
from sensor_msgs.msg import NavSatFix
from collections import defaultdict
import os
from ublox_msgs.msg import NavPVT

def decode_fix_mode(navpvt_msg):
    """
    Decodes diffSoln and carrSoln from flags and returns a numeric code.
    Mapping:
        1 = NO_FIX
        2 = AUTONOMOUS
        3 = DGNSS
        4 = RTK_FLOAT
        5 = RTK_FIXED
        0 = UNKNOWN
    """
    flags = navpvt_msg.flags
    fix_type = navpvt_msg.fix_type

    diffSoln = (flags >> 1) & 1      # bit 1 → differential
    carrSoln = (flags >> 6) & 0x3    # bits 6–7 → carrier solution

    if fix_type < 3:
        return 1  # NO_FIX
    elif diffSoln == 0 and carrSoln == 0:
        return 2  # AUTONOMOUS
    elif diffSoln == 1 and carrSoln == 0:
        return 3  # DGNSS
    elif diffSoln == 1 and carrSoln == 1:
        return 4  # RTK_FLOAT
    elif diffSoln == 1 and carrSoln == 2:
        return 5  # RTK_FIXED
    else:
        return 0  # UNKNOWN



def main():
    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Converts NavSatFix topics from a rosbag into GPX files"
    )
    parser.add_argument(
        "--bag_path",
        required=True,
        help="Path to the rosbag (directory)"
    )
    parser.add_argument(
        "--mode",
        choices=["altitude", "fixcode"],
        default="altitude",
        help="Output mode: altitude (uses real altitude) or fixcode (replaces altitude with correction code)"
    )
    args = parser.parse_args()

    # Open rosbag
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    # Define NavSatFix topics
    navsat_topics = [
        '/PREMOVE/left/gps/fix',
        '/PREMOVE/right/gps/fix'
    ]

    navpvt_topics = [
        '/PREMOVE/left/gps/navpvt',
        '/PREMOVE/right/gps/navpvt'
    ]

    coordinates_by_topic = defaultdict(list)

    last_fixcode_by_side = {"left": -1, "right": -1}

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic in navpvt_topics:
            navpvt_msg = rclpy.serialization.deserialize_message(data, NavPVT)
            side = "left" if "left" in topic else "right"
            last_fixcode_by_side[side] = decode_fix_mode(navpvt_msg)

        elif topic in navsat_topics:
            fix_msg = rclpy.serialization.deserialize_message(data, NavSatFix)
            lat, lon = fix_msg.latitude, fix_msg.longitude
            if args.mode == "altitude":
                value = fix_msg.altitude
            else:
                side = "left" if "left" in topic else "right"
                value = last_fixcode_by_side[side]

            stamp_sec = fix_msg.header.stamp.sec
            stamp_nanosec = fix_msg.header.stamp.nanosec
            timestamp = stamp_sec + stamp_nanosec * 1e-9
            time_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

            coordinates_by_topic[topic].append((lat, lon, value, time_str))

    # Creates a GPX file for each topic. The name is composed of bag_name+topic+fix_mode
    bag_basename = os.path.basename(os.path.normpath(args.bag_path))
    mode_suffix = "position" if args.mode == "altitude" else "fixcode"

    for topic_name, coords in coordinates_by_topic.items():
        topic_suffix = topic_name.strip("/").replace("/", "_")
        filename = f"{bag_basename}_{topic_suffix}_{mode_suffix}.gpx"

        gpx = ET.Element('gpx', {
            'version': '1.1',
            'creator': f'ROS2 to GPX Converter ({args.mode})',
            'xmlns': 'http://www.topografix.com/GPX/1/1',
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:schemaLocation': 'http://www.topografix.com/GPX/1/1/gpx.xsd'
        })

        trk = ET.SubElement(gpx, 'trk')
        ET.SubElement(trk, 'name').text = topic_suffix
        trkseg = ET.SubElement(trk, 'trkseg')

        for lat, lon, value, time_str in coords:
            trkpt = ET.SubElement(trkseg, 'trkpt', attrib={'lat': str(lat), 'lon': str(lon)})
            ET.SubElement(trkpt, 'ele').text = str(value)
            ET.SubElement(trkpt, 'time').text = time_str

        ET.ElementTree(gpx).write(filename, encoding='utf-8', xml_declaration=True)
        print(f"✅ GPX generated: {filename}")


if __name__ == "__main__":
    main()
