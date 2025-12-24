import argparse
import sys
import rosbag2_py


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trim a ROS 2 bag into time windows while preserving original timestamps"
        )
    )
    parser.add_argument("input_bag", help="Path to the input rosbag")
    parser.add_argument("output_bag", help="Path to the output rosbag")
    parser.add_argument(
        "--ranges",
        type=float,
        nargs="+",
        required=True,
        help="t1 t2 t3 t4 ... time ranges in seconds from bag start"
    )
    parser.add_argument(
        "--storage",
        default="sqlite3",
        help="Storage backend: sqlite3 or mcap"
    )

    args = parser.parse_args()

    # Ranges must be provided as start/end pairs
    if len(args.ranges) % 2 != 0:
        print("--ranges must contain start/end pairs")
        sys.exit(1)

    # Convert time ranges from seconds to nanoseconds
    ranges_ns = []
    for i in range(0, len(args.ranges), 2):
        start = int(args.ranges[i] * 1e9)
        end = int(args.ranges[i + 1] * 1e9)
        if end <= start:
            raise RuntimeError("Invalid time range")
        ranges_ns.append((start, end))

    # Initialize reader
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(args.input_bag, args.storage),
        rosbag2_py.ConverterOptions("", "")
    )

    # Read first message to get the actual bag start time
    if not reader.has_next():
        print("The bag is empty")
        sys.exit(1)

    first_topic, first_data, first_timestamp = reader.read_next()
    bag_start = first_timestamp

    # Initialize writer
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(args.output_bag, args.storage),
        rosbag2_py.ConverterOptions("", "")
    )

    # Register all topics and types
    for topic in reader.get_all_topics_and_types():
        writer.create_topic(topic)

    # Process first message
    rel_time = first_timestamp - bag_start
    for start, end in ranges_ns:
        if start <= rel_time <= end:
            writer.write(first_topic, first_data, first_timestamp)
            break

    print("▶️ Processing messages...")

    # Process remaining messages
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        rel_time = timestamp - bag_start

        for start, end in ranges_ns:
            if start <= rel_time <= end:
                writer.write(topic, data, timestamp)
                break

    print(" Bag successfully generated:", args.output_bag)


if __name__ == "__main__":
    main()
