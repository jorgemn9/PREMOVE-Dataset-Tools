# PREMOVE-Dataset-Tools
Set of tools to facilitate data extraction, conversion, and processing from the original recordings of the PREMOVE dataset

## Data acquisition

### Obtain car positions in GPX format. Two visualization modes are provided: a geographic mode displaying latitude, longitude, and altitude profiles, and an RTK quality mode that replaces altitude with fix status

``python3 rosbag2_gnss_to_gpx.py --bag_path $your_bag_path$ --mode "altitude"``

``python3 rosbag2_gnss_to_gpx.py --bag_path $your_bag_path$ --mode "fixcode"``

### Extract LiDAR frames and export them as raw 3D arrays in npz format and as BEV projections in png format. Parameter "topic_name" in the script must be switched between /rslidar_points and /rslidar_points_2

``python3 rosbag2_pointcloud_processor.py $your_bag_path$``

### Extract thermal images and saves it in jpg format
``python3 rosbag2_thermal_to_jpg.py $your_bag_path$``

### Extract left-channel stereo images and saves it in jpg format
``source ~/ros2_ws/install/setup.bash``

``python3 stereo_image_saver.py``

``ros2 launch zed_multi_camera zed_multi_camera_svo.launch.py cam_names:="[stereo_$number$]" cam_models:="[zed2i]" cam_serials:="[$your_cam_serial$]" svo_paths:="[$your_svo_path$]"``

### Record IMU measurement in a rosbag2 file and save it in a single YAML file. Then save IMU data in YAML files
``source ~/ros2_ws/install/setup.bash``

``ros2 launch zed_multi_camera zed_multi_camera_svo.launch.py cam_names:="[stereo_$number$]" cam_models:="[zed2i]" cam_serials:="[$your_cam_serial$]" svo_paths:="[$your_svo_path$]"``

``ros2 bag record \
  /zed_multi/stereo_$number$/left_cam_imu_transform \
  /zed_multi/stereo_$number$/imu/data \
  -o IMU_Bags/stereo$number$_imu``

``python3 rosbag2_imu_saver.py``

## Data synchronization
### Images and point clouds synchronization
``python3 data_synchronizer.py $path_to_dataset_folder$ --tolerance-ms 60 --link copy --format yaml --make-collages --lidar-source npz --ref-sensor zed_multi_stereo_2_rgb_image_rect_color``

### GNSS synchronization
``python3 extract_gnss.py $path_to_gnss_bag$``

## Create dataset in nuScenes-like format
``python3 create_dataset.py``

## Blur plates and faces (Optional)
``cd faces_plates_anonymization``

``python3 faces_plates_anonymization.py --source $dataset_folder$ --out $blur_dataset_folder$ --face-weights ./weights/yolo11n_faces.pt --plate-weights ./weights/yolo11n_licenseplates.pt --overwrite``

## Create new collage from censored data
``python3 collage.py``

## Create video from collage images
``ffmpeg -framerate 30 -pattern_type glob -i "collages_censored/collage-*.jpg" -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 28 collage_video.mp4``

