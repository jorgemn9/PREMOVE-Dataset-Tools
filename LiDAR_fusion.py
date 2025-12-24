import numpy as np
import open3d as o3d
import argparse

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (in radians) to a rotation matrix R = Rz * Ry * Rx
    """
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ])

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ])

    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def transform_points(points, t, r):
    """
    Apply an extrinsic transform to a point cloud.

    points: Nx3 array
    t: translation [x, y, z]
    r: rotation [roll, pitch, yaw] in radians
    """
    R = euler_to_rotation_matrix(*r)
    T = np.array(t).reshape(1, 3)
    return (R @ points.T).T + T

def numpy_to_o3d(p):
    """
    Converts an Nx3 array to an Open3D point cloud
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(p)
    return pc

# Parse point cloud paths from command line
parser = argparse.ArgumentParser(description="Visualize two LiDAR point clouds")
parser.add_argument("--pc1", required=True, help="Path to LiDAR 1 point cloud (.npz)")
parser.add_argument("--pc2", required=True, help="Path to LiDAR 2 point cloud (.npz)")
args = parser.parse_args()

# Loads both LiDAR point clouds
pc1 = np.load(args.pc1)["points"][:, :3]   # (N,3)
pc2 = np.load(args.pc2)["points"][:, :3]   # (M,3)


# Define extrinsic parameters between both LiDAR sensors
# LiDAR_1 translation and rotation w.r.t the origin of the platform
lid_t  = [-5.02, -7.60, 466.59]
lid_r  = [0, 0, 0]

# LiDAR_2 translation and rotation w.r.t the origin of the platform
lid2_t = [0.52, -1.73, 620.88]
lid2_r = [0, 1, 0]

# Convert translation vectors from millimetres to metres
lid_t  = [v / 1000.0 for v in lid_t]
lid2_t = [v / 1000.0 for v in lid2_t]

# Convert rotation vectors from degrees to radians
lid_r  = [np.deg2rad(v) for v in lid_r]
lid2_r = [np.deg2rad(v) for v in lid2_r]

# Transform points from each reference frame to the origin of the platform one
pc1_global = transform_points(pc1, lid_t, lid_r)
pc2_global = transform_points(pc2, lid2_t, lid2_r)

# Transform points from Numpy to Open3D format
pc1_o3d = numpy_to_o3d(pc1_global)
pc2_o3d = numpy_to_o3d(pc2_global)

# Defines the color of each point cloud
pc1_o3d.paint_uniform_color([1, 0, 0])   # red
pc2_o3d.paint_uniform_color([0, 0, 1])   # blue

# Creates the Open3D visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Nubes LIDAR", width=3840, height=2160)

vis.add_geometry(pc1_o3d)
vis.add_geometry(pc2_o3d)

opt = vis.get_render_option()
opt.point_size = 3        
opt.background_color = np.array([1, 1, 1]) 

vis.run()
vis.destroy_window()
