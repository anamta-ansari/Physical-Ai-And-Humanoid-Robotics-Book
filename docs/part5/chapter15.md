---
title: Vision Systems for Humanoid Robots
sidebar_position: 4
description: Computer vision, depth sensing, object detection, tracking, SLAM, and visual servoing for humanoid robots
---

# Vision Systems for Humanoid Robots

## Camera systems and calibration

Computer vision is crucial for humanoid robots to perceive and understand their environment. This chapter covers the implementation of vision systems specifically tailored for humanoid robotics applications, including camera selection, calibration, and advanced vision algorithms.

### Camera Selection for Humanoid Robots

Humanoid robots require specialized camera systems that can handle the dynamic nature of bipedal locomotion and provide the necessary information for navigation, manipulation, and interaction.

```python
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
import yaml

class CameraSelection:
    """
    Class to help select appropriate cameras for humanoid robots based on requirements
    """
    def __init__(self):
        # Define camera types with their specifications
        self.camera_specs = {
            'monochrome_global_shutter': {
                'resolution': (640, 480),
                'frame_rate': 120,
                'shutter_type': 'global',
                'latency': 0.002,  # 2ms
                'light_sensitivity': 0.5,  # lux
                'dynamic_range': 60,  # dB
                'power_consumption': 1.2,  # watts
                'weight': 0.03,  # kg
                'cost': 150,  # USD
                'use_cases': ['motion_capture', 'high_speed tracking', 'low light']
            },
            'rgb_global_shutter': {
                'resolution': (1280, 720),
                'frame_rate': 60,
                'shutter_type': 'global',
                'latency': 0.003,
                'light_sensitivity': 1.0,
                'dynamic_range': 65,
                'power_consumption': 2.0,
                'weight': 0.05,
                'cost': 300,
                'use_cases': ['color tracking', 'object recognition', 'navigation']
            },
            'stereo_vision': {
                'resolution': (640, 480),
                'frame_rate': 30,
                'shutter_type': 'global',
                'latency': 0.01,
                'light_sensitivity': 2.0,
                'dynamic_range': 70,
                'power_consumption': 3.5,
                'weight': 0.1,
                'cost': 500,
                'use_cases': ['depth_estimation', '3D reconstruction', 'obstacle_avoidance']
            },
            'event_camera': {
                'resolution': (640, 480),
                'frame_rate': 1000,  # Extremely high due to event-based nature
                'shutter_type': 'event',
                'latency': 0.00001,  # Extremely low
                'light_sensitivity': 0.01,
                'dynamic_range': 120,
                'power_consumption': 0.8,
                'weight': 0.04,
                'cost': 1000,
                'use_cases': ['high-speed motion', 'dynamic scenes', 'low latency']
            },
            'thermal_camera': {
                'resolution': (384, 288),
                'frame_rate': 30,
                'shutter_type': 'rolling',  # Most thermal cameras use rolling shutter
                'latency': 0.03,
                'light_sensitivity': 'n/a',  # Works in darkness
                'dynamic_range': 'n/a',  # Different concept for thermal
                'power_consumption': 2.5,
                'weight': 0.15,
                'cost': 4000,
                'use_cases': ['night_vision', 'temperature_monitoring', 'hazard_detection']
            }
        }
    
    def recommend_cameras(self, robot_specifications: Dict) -> List[str]:
        """
        Recommend camera types based on robot specifications
        """
        recommendations = []
        
        # Consider robot size and weight constraints
        max_camera_weight = robot_specifications.get('max_camera_weight', 0.2)  # kg
        max_power_draw = robot_specifications.get('max_camera_power', 10.0)  # watts
        
        # Consider intended applications
        applications = robot_specifications.get('applications', [])
        
        for cam_type, spec in self.camera_specs.items():
            # Check weight and power constraints
            if spec['weight'] > max_camera_weight or spec['power_consumption'] > max_power_draw:
                continue
            
            # Check if camera type matches any of the applications
            if any(app in spec['use_cases'] for app in applications):
                recommendations.append(cam_type)
        
        return recommendations

# Example usage
camera_selector = CameraSelection()

# Define robot specifications
robot_spec = {
    'max_camera_weight': 0.15,  # kg
    'max_camera_power': 8.0,    # watts
    'applications': ['navigation', 'object_recognition', 'obstacle_avoidance']
}

recommended_cameras = camera_selector.recommend_cameras(robot_spec)
print(f"Recommended cameras: {recommended_cameras}")
```

### Camera Calibration

Proper camera calibration is essential for accurate computer vision applications. Calibration involves determining both intrinsic and extrinsic parameters.

```python
class CameraCalibrator:
    """
    Class for camera calibration using chessboard patterns
    """
    def __init__(self):
        self.intrinsic_matrix = None
        self.distortion_coeffs = None
        self.extrinsic_matrix = None
    
    def calibrate_camera(self, images: List[np.ndarray], board_shape: Tuple[int, int], 
                        square_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate camera using a series of images with calibration pattern
        """
        # Prepare object points (3D points in real world space)
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane
        
        # Create 3D points for the chessboard corners
        objp = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_size
        
        # Find corners in each image
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, board_shape, None)
            
            if ret:
                obj_points.append(objp)
                # Refine corner locations
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                img_points.append(refined_corners)
        
        if len(obj_points) == 0:
            raise ValueError("No chessboard corners found in any of the images")
        
        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray.shape[::-1], None, None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        self.intrinsic_matrix = mtx
        self.distortion_coeffs = dist
        
        print(f"Calibration successful!")
        print(f"Intrinsic matrix:\n{mtx}")
        print(f"Distortion coefficients:\n{dist}")
        print(f"Reprojection error: {ret}")
        
        return mtx, dist
    
    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort an image using the calibrated parameters
        """
        if self.intrinsic_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            self.intrinsic_matrix, self.distortion_coeffs, (w, h), 1, (w, h)
        )
        
        # Undistort
        dst = cv2.undistort(img, self.intrinsic_matrix, self.distortion_coeffs, None, new_camera_mtx)
        
        # Crop the image based on ROI
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def stereo_calibrate(self, left_images: List[np.ndarray], right_images: List[np.ndarray],
                        board_shape: Tuple[int, int], square_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrate stereo camera system
        """
        # Prepare object points
        obj_points = []
        left_img_points = []
        right_img_points = []
        
        objp = np.zeros((board_shape[0] * board_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_size
        
        # Find corners in both cameras
        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Find corners in left image
            ret_left, left_corners = cv2.findChessboardCorners(left_gray, board_shape, None)
            if ret_left:
                left_refined = cv2.cornerSubPix(
                    left_gray, left_corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
            else:
                continue
            
            # Find corners in right image
            ret_right, right_corners = cv2.findChessboardCorners(right_gray, board_shape, None)
            if ret_right:
                right_refined = cv2.cornerSubPix(
                    right_gray, right_corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
            else:
                continue
            
            # Only use if both images have detected corners
            if ret_left and ret_right:
                obj_points.append(objp)
                left_img_points.append(left_refined)
                right_img_points.append(right_refined)
        
        if len(obj_points) < 3:
            raise ValueError("Need at least 3 images with valid corner detections for stereo calibration")
        
        # Calibrate individual cameras first
        ret_left, ml, dl, rl, tl = cv2.calibrateCamera(obj_points, left_img_points, left_gray.shape[::-1], None, None)
        ret_right, mr, dr, rr, tr = cv2.calibrateCamera(obj_points, right_img_points, right_gray.shape[::-1], None, None)
        
        # Stereo calibration
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, ml, dl, mr, dr, R, T, E, F = cv2.stereoCalibrate(
            obj_points, left_img_points, right_img_points,
            ml, dl, mr, dr,
            left_gray.shape[::-1], criteria=stereocalib_criteria,
            flags=flags
        )
        
        if not ret:
            raise RuntimeError("Stereo calibration failed")
        
        # Compute rectification parameters
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roi_l, roi_r = cv2.stereoRectify(
            ml, dl, mr, dr, left_gray.shape[::-1], R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.stereo_rectification_params = {
            'rect_l': rect_l,
            'rect_r': rect_r,
            'proj_mat_l': proj_mat_l,
            'proj_mat_r': proj_mat_r,
            'disparity_to_depth_map': Q
        }
        
        print(f"Stereo calibration successful!")
        print(f"Translation vector: {T.flatten()}")
        print(f"Rotation matrix:\n{R}")
        print(f"Essential matrix:\n{E}")
        print(f"Fundamental matrix:\n{F}")
        
        return R, T, Q

# Example usage
calibrator = CameraCalibrator()

# Example of loading and calibrating from images (in practice, you'd load actual calibration images)
# For this example, we'll create dummy images
dummy_images = [np.random.rand(480, 640, 3) * 255 for _ in range(10)]
dummy_images = [img.astype(np.uint8) for img in dummy_images]

# Calibrate camera (this would normally use real chessboard images)
# intrinsic_matrix, distortion_coeffs = calibrator.calibrate_camera(dummy_images, (9, 6), 25.0)
```

### Multiple Camera Integration

Humanoid robots often have multiple cameras positioned differently on the body. Proper integration of these cameras is essential for complete environmental perception.

```python
class MultiCameraSystem:
    """
    Manages multiple cameras on a humanoid robot
    """
    def __init__(self):
        self.cameras = {}  # Dictionary to store camera information
        self.camera_poses = {}  # Transform from robot base to each camera
        self.camera_calibrations = {}  # Calibration parameters for each camera
    
    def add_camera(self, name: str, camera_info: Dict, pose: np.ndarray):
        """
        Add a camera to the system
        camera_info: Dictionary with camera parameters
        pose: 4x4 transformation matrix from robot base to camera
        """
        self.cameras[name] = camera_info
        self.camera_poses[name] = pose
    
    def get_camera_extrinsics(self, camera_name: str) -> np.ndarray:
        """
        Get the extrinsic parameters (pose) of a specific camera
        """
        if camera_name not in self.camera_poses:
            raise ValueError(f"Camera {camera_name} not found")
        return self.camera_poses[camera_name]
    
    def get_camera_intrinsics(self, camera_name: str) -> Dict:
        """
        Get the intrinsic parameters of a specific camera
        """
        if camera_name not in self.cameras:
            raise ValueError(f"Camera {camera_name} not found")
        return self.cameras[camera_name]['intrinsic_matrix']
    
    def transform_point_to_robot_frame(self, camera_name: str, point_2d: np.ndarray, depth: float) -> np.ndarray:
        """
        Transform a 2D point with depth from camera frame to robot base frame
        """
        # First, convert 2D point + depth to 3D in camera frame
        intrinsic = self.get_camera_intrinsics(camera_name)
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        
        # Convert to 3D point in camera frame
        x_cam = (point_2d[0] - cx) * depth / fx
        y_cam = (point_2d[1] - cy) * depth / fy
        z_cam = depth
        
        point_cam_frame = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # Transform to robot base frame
        extrinsic = self.get_camera_extrinsics(camera_name)
        point_robot_frame = extrinsic @ point_cam_frame
        
        return point_robot_frame[:3]  # Return 3D position
    
    def fuse_camera_data(self, camera_data: Dict[str, np.ndarray]) -> Dict:
        """
        Fuse data from multiple cameras to create a unified perception
        """
        fused_data = {
            'point_cloud': [],
            'detected_objects': [],
            'free_space': [],
            'obstacles': []
        }
        
        for camera_name, data in camera_data.items():
            # Transform camera data to robot base frame
            camera_extrinsics = self.get_camera_extrinsics(camera_name)
            
            if 'point_cloud' in data:
                # Transform point cloud to robot frame
                points_robot_frame = []
                for point in data['point_cloud']:
                    point_homogeneous = np.append(point, 1.0)
                    point_robot = camera_extrinsics @ point_homogeneous
                    points_robot_frame.append(point_robot[:3])
                
                fused_data['point_cloud'].extend(points_robot_frame)
            
            if 'detected_objects' in data:
                # Transform object positions to robot frame
                for obj in data['detected_objects']:
                    obj_pos_cam = np.append(obj['position'], 1.0)
                    obj_pos_robot = camera_extrinsics @ obj_pos_cam
                    
                    obj_robot_frame = obj.copy()
                    obj_robot_frame['position'] = obj_pos_robot[:3]
                    obj_robot_frame['camera_source'] = camera_name
                    fused_data['detected_objects'].append(obj_robot_frame)
        
        return fused_data

# Example setup for a humanoid robot with multiple cameras
multi_camera_system = MultiCameraSystem()

# Add cameras to the robot
# Head camera (looking forward)
head_camera_pose = np.array([
    [1, 0, 0, 0.0],      # 0m forward
    [0, 1, 0, 0.0],      # 0m left/right
    [0, 0, 1, 1.7],      # 1.7m high (head height)
    [0, 0, 0, 1]
])

multi_camera_system.add_camera(
    'head_camera',
    {
        'intrinsic_matrix': np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ]),
        'distortion_coeffs': np.array([0.1, -0.2, 0, 0, 0.1]),
        'resolution': (640, 480)
    },
    head_camera_pose
)

# Chest camera (looking down at hands/ground)
chest_camera_pose = np.array([
    [1, 0, 0, 0.0],      # 0m forward
    [0, 1, 0, 0.0],      # 0m left/right
    [0, 0, 1, 1.2],      # 1.2m high (chest height)
    [0, 0, 0, 1]
])

multi_camera_system.add_camera(
    'chest_camera',
    {
        'intrinsic_matrix': np.array([
            [520, 0, 320],
            [0, 520, 240],
            [0, 0, 1]
        ]),
        'distortion_coeffs': np.array([0.05, -0.1, 0, 0, 0.05]),
        'resolution': (640, 480)
    },
    chest_camera_pose
)

# Stereo cameras on head
left_eye_pose = np.array([
    [1, 0, 0, 0.05],     # 5cm to the right
    [0, 1, 0, 0.06],     # 6cm forward
    [0, 0, 1, 1.7],      # 1.7m high
    [0, 0, 0, 1]
])

right_eye_pose = np.array([
    [1, 0, 0, 0.05],     # 5cm to the right
    [0, 1, 0, -0.06],    # 6cm backward
    [0, 0, 1, 1.7],      # 1.7m high
    [0, 0, 0, 1]
])

multi_camera_system.add_camera(
    'left_eye',
    {
        'intrinsic_matrix': np.array([
            [480, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ]),
        'distortion_coeffs': np.array([0.08, -0.15, 0, 0, 0.08]),
        'resolution': (640, 480)
    },
    left_eye_pose
)

multi_camera_system.add_camera(
    'right_eye',
    {
        'intrinsic_matrix': np.array([
            [480, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ]),
        'distortion_coeffs': np.array([0.08, -0.15, 0, 0, 0.08]),
        'resolution': (640, 480)
    },
    right_eye_pose
)

print("Multi-camera system initialized with 4 cameras")
print(f"Available cameras: {list(multi_camera_system.cameras.keys())}")
```

## Depth sensing and 3D reconstruction

Depth sensing is crucial for humanoid robots to understand the 3D structure of their environment, enabling navigation, manipulation, and interaction.

### Depth Camera Integration

```python
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class DepthCameraProcessor:
    """
    Class for processing depth camera data for humanoid robots
    """
    def __init__(self, camera_params: Dict):
        """
        Initialize with camera parameters
        """
        self.fx = camera_params['fx']
        self.fy = camera_params['fy']
        self.cx = camera_params['cx']
        self.cy = camera_params['cy']
        self.width = camera_params['width']
        self.height = camera_params['height']
        
        # Filtering parameters
        self.depth_min = camera_params.get('depth_min', 0.1)  # meters
        self.depth_max = camera_params.get('depth_max', 5.0)  # meters
        self.spatial_filter_radius = 0.03  # meters
        self.temporal_filter_alpha = 0.1  # For temporal smoothing
    
    def depth_to_point_cloud(self, depth_image: np.ndarray, color_image: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """
        Convert depth image to point cloud
        """
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        
        # Convert pixel coordinates to camera coordinates
        x_cam = (x_coords - self.cx) * depth_image / self.fx
        y_cam = (y_coords - self.cy) * depth_image / self.fy
        
        # Stack coordinates
        points = np.stack([x_cam, y_cam, depth_image], axis=-1).reshape(-1, 3)
        
        # Remove invalid points (where depth is 0 or outside range)
        valid_mask = (depth_image > self.depth_min) & (depth_image < self.depth_max) & (depth_image > 0)
        valid_points = points[valid_mask.flatten()]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        
        if color_image is not None:
            # Add color information
            valid_colors = color_image.reshape(-1, 3)[valid_mask.flatten()]
            pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)  # Normalize to [0,1]
        
        return pcd
    
    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Apply various filters to the point cloud
        """
        # Remove statistical outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_filtered = pcd.select_by_index(ind)
        
        # Apply radius outlier removal
        cl, ind = pcd_filtered.remove_radius_outlier(nb_points=16, radius=0.05)
        pcd_filtered = pcd_filtered.select_by_index(ind)
        
        # Downsample for efficiency
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=0.01)  # 1cm voxels
        
        return pcd_downsampled
    
    def create_occupancy_grid(self, pcd: o3d.geometry.PointCloud, resolution: float = 0.1) -> Dict:
        """
        Create an occupancy grid from point cloud data
        """
        points = np.asarray(pcd.points)
        
        # Determine grid bounds
        min_bounds = np.floor(np.min(points, axis=0) / resolution) * resolution
        max_bounds = np.ceil(np.max(points, axis=0) / resolution) * resolution
        
        # Create grid
        grid_size = ((max_bounds - min_bounds) / resolution).astype(int)
        occupancy_grid = np.zeros(grid_size, dtype=np.uint8)
        
        # Populate grid
        grid_indices = ((points - min_bounds) / resolution).astype(int)
        
        # Clamp indices to valid range
        grid_indices = np.clip(grid_indices, 
                              np.array([0, 0, 0]), 
                              np.array(grid_size) - 1)
        
        # Mark occupied cells
        for idx in grid_indices:
            occupancy_grid[idx[0], idx[1], idx[2]] = 1
        
        return {
            'grid': occupancy_grid,
            'resolution': resolution,
            'origin': min_bounds,
            'size': grid_size
        }
    
    def segment_ground_plane(self, pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.05) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Segment ground plane from point cloud using RANSAC
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        ground_cloud = pcd.select_by_index(inliers)
        obstacle_cloud = pcd.select_by_index(inliers, invert=True)
        
        return ground_cloud, obstacle_cloud
    
    def extract_planes(self, pcd: o3d.geometry.PointCloud, min_plane_size: int = 100) -> List[o3d.geometry.PointCloud]:
        """
        Extract multiple planes from the point cloud
        """
        planes = []
        remaining_pcd = pcd
        
        while len(remaining_pcd.points) > min_plane_size:
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < min_plane_size:
                break  # Not enough points for a valid plane
            
            plane_cloud = remaining_pcd.select_by_index(inliers)
            planes.append(plane_cloud)
            
            # Remove this plane from remaining points
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
        return planes

# Example usage with a simulated depth camera
depth_params = {
    'fx': 500, 'fy': 500, 'cx': 320, 'cy': 240,
    'width': 640, 'height': 480,
    'depth_min': 0.1, 'depth_max': 5.0
}

depth_processor = DepthCameraProcessor(depth_params)

# Example of how to use with real data (simulated here)
# In practice, you'd get depth and color images from your depth camera
simulated_depth = np.random.rand(480, 640) * 4.0 + 0.1  # 0.1 to 4.1 meters
simulated_color = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Create point cloud from depth
point_cloud = depth_processor.depth_to_point_cloud(simulated_depth, simulated_color)

# Filter the point cloud
filtered_pcd = depth_processor.filter_point_cloud(point_cloud)

# Segment ground plane
ground_cloud, obstacle_cloud = depth_processor.segment_ground_plane(filtered_pcd)

print(f"Original point cloud: {len(point_cloud.points)} points")
print(f"Filtered point cloud: {len(filtered_pcd.points)} points")
print(f"Ground points: {len(ground_cloud.points)} points")
print(f"Obstacle points: {len(obstacle_cloud.points)} points")
```

### 3D Reconstruction Pipeline

```python
class ThreeDReconstructionPipeline:
    """
    Complete pipeline for 3D reconstruction from depth data
    """
    def __init__(self, camera_params: Dict):
        self.depth_processor = DepthCameraProcessor(camera_params)
        self.mesh_resolution = 0.02  # 2cm resolution for mesh generation
        self.surface_density = 3  # Minimum points per surface element
    
    def reconstruct_scene(self, depth_frames: List[np.ndarray], 
                         color_frames: List[np.ndarray],
                         camera_poses: List[np.ndarray]) -> o3d.geometry.TriangleMesh:
        """
        Reconstruct 3D scene from multiple depth frames with known poses
        """
        # Initialize TSDF volume for fusion
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02,  # 2cm voxels
            sdf_trunc=0.04,     # Truncated distance value
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Integrate each frame
        for depth_img, color_img, pose in zip(depth_frames, color_frames, camera_poses):
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_img.astype(np.uint8)),
                o3d.geometry.Image(depth_img.astype(np.float32)),
                depth_scale=1.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False
            )
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                self.depth_processor.width, 
                self.depth_processor.height,
                self.depth_processor.fx, 
                self.depth_processor.fy,
                self.depth_processor.cx, 
                self.depth_processor.cy
            )
            
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        
        # Extract mesh
        mesh = volume.extract_triangle_mesh()
        
        # Clean up the mesh
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_vertex_normals()
        
        return mesh
    
    def extract_objects(self, reconstructed_scene: o3d.geometry.TriangleMesh) -> List[o3d.geometry.TriangleMesh]:
        """
        Extract individual objects from the reconstructed scene
        """
        # Convert mesh to point cloud for clustering
        pcd = reconstructed_scene.sample_points_uniformly(number_of_points=10000)
        
        # Perform clustering to identify separate objects
        labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=100, print_progress=False))
        
        objects = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue
            
            # Get points belonging to this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            
            # Create bounding box for the cluster
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            
            # Optionally, create a mesh for this object
            # This is a simplified approach - in practice you might want to use surface reconstruction
            object_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            object_mesh.translate(bbox.get_center())
            
            objects.append({
                'mesh': object_mesh,
                'bbox': bbox,
                'center': bbox.get_center(),
                'size': bbox.get_extent()
            })
        
        return objects
    
    def generate_navigation_map(self, reconstructed_scene: o3d.geometry.TriangleMesh, 
                               robot_radius: float = 0.3) -> np.ndarray:
        """
        Generate 2D navigation map from 3D reconstruction
        """
        # Convert to point cloud for processing
        pcd = reconstructed_scene.sample_points_uniformly(number_of_points=50000)
        
        # Project to 2D grid
        points = np.asarray(pcd.points)
        
        # Create 2D occupancy grid
        resolution = 0.05  # 5cm resolution
        min_x, min_y = np.min(points[:, :2], axis=0) - robot_radius
        max_x, max_y = np.max(points[:, :2], axis=0) + robot_radius
        
        width = int((max_x - min_x) / resolution)
        height = int((max_y - min_y) / resolution)
        
        occupancy_map = np.zeros((height, width), dtype=np.uint8)
        
        # Project 3D points to 2D grid
        for point in points:
            if point[2] < 0.5:  # Only consider ground-level obstacles
                x_idx = int((point[0] - min_x) / resolution)
                y_idx = int((point[1] - min_y) / resolution)
                
                if 0 <= x_idx < width and 0 <= y_idx < height:
                    occupancy_map[y_idx, x_idx] = 1  # Occupied
        
        # Apply robot radius to dilate obstacles
        from scipy.ndimage import binary_dilation
        structure = np.ones((int(2*robot_radius/resolution), int(2*robot_radius/resolution)))
        dilated_map = binary_dilation(occupancy_map, structure=structure).astype(np.uint8)
        
        return dilated_map, (min_x, min_y), resolution

# Example usage
reconstruction_pipeline = ThreeDReconstructionPipeline(depth_params)

# Simulate multiple frames with poses (in practice, these would come from visual odometry or SLAM)
num_frames = 10
simulated_depth_frames = [np.random.rand(480, 640) * 3.0 + 0.5 for _ in range(num_frames)]
simulated_color_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]

# Simulate camera poses (moving in a circle)
poses = []
for i in range(num_frames):
    angle = 2 * np.pi * i / num_frames
    x = 0.5 * np.cos(angle)
    y = 0.5 * np.sin(angle)
    z = 1.0
    
    # Simple rotation to look at center
    yaw = np.arctan2(-y, -x)
    
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler('z', yaw).as_matrix()
    pose[:3, 3] = [x, y, z]
    poses.append(pose)

# Perform 3D reconstruction
# reconstructed_mesh = reconstruction_pipeline.reconstruct_scene(
#     simulated_depth_frames, simulated_color_frames, poses
# )
# 
# print(f"Reconstructed mesh with {len(reconstructed_mesh.vertices)} vertices and {len(reconstructed_mesh.triangles)} triangles")
```

## Object detection and tracking

Object detection and tracking are critical capabilities for humanoid robots to interact with their environment.

### Object Detection Pipeline

```python
import torch
import torchvision
from torchvision import transforms
import cv2
from ultralytics import YOLO

class ObjectDetectionSystem:
    """
    Object detection system for humanoid robots using deep learning
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize object detection system
        """
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Common classes relevant to humanoid robots
        self.relevant_classes = {
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        }
        
        # Confidence threshold
        self.confidence_threshold = 0.5
        
        # IOU threshold for NMS
        self.iou_threshold = 0.5
        
        # Class mapping for humanoid-specific tasks
        self.task_mappings = {
            'graspable_objects': ['bottle', 'cup', 'book', 'laptop', 'cell phone', 
                                 'backpack', 'handbag', 'suitcase', 'bowl', 'fork', 
                                 'knife', 'spoon', 'plate', 'toy', 'teddy bear'],
            'navigation_obstacles': ['person', 'chair', 'couch', 'table', 'plant', 
                                   'bench', 'bicycle', 'car', 'truck'],
            'social_entities': ['person', 'dog', 'cat'],
            'household_items': ['refrigerator', 'microwave', 'oven', 'sink', 
                              'bed', 'toilet', 'couch', 'chair', 'table']
        }
    
    def detect_objects(self, image: np.ndarray, task_type: str = 'general') -> List[Dict]:
        """
        Detect objects in an image
        """
        # Run YOLO detection
        results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            names = result.names  # Class names dictionary
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                class_name = names[int(class_id)]
                
                # Only include relevant classes
                if class_name in self.relevant_classes:
                    # For specific tasks, filter further
                    if task_type != 'general':
                        if task_type in self.task_mappings:
                            if class_name not in self.task_mappings[task_type]:
                                continue
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x) for x in box],  # x1, y1, x2, y2
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],  # Center of bounding box
                        'area': (box[2] - box[0]) * (box[3] - box[1])  # Area of bounding box
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def filter_detections_by_task(self, detections: List[Dict], task_type: str) -> List[Dict]:
        """
        Filter detections based on task requirements
        """
        if task_type == 'grasping':
            return [det for det in detections if det['class_name'] in self.task_mappings['graspable_objects']]
        elif task_type == 'navigation':
            return [det for det in detections if det['class_name'] in self.task_mappings['navigation_obstacles']]
        elif task_type == 'social_interaction':
            return [det for det in detections if det['class_name'] in self.task_mappings['social_entities']]
        else:
            return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       task_type: str = 'general') -> np.ndarray:
        """
        Draw detection results on image
        """
        output_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            
            # Choose color based on task type
            if task_type == 'grasping':
                color = (0, 255, 0)  # Green for graspable objects
            elif task_type == 'navigation':
                color = (0, 0, 255)  # Red for navigation obstacles
            elif task_type == 'social_interaction':
                color = (255, 0, 0)  # Blue for social entities
            else:
                color = (255, 255, 0)  # Yellow for general detections
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(output_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image

# Example usage
detector = ObjectDetectionSystem()

# Example detection (using simulated image)
simulated_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
detections = detector.detect_objects(simulated_image, task_type='grasping')

print(f"Detected {len(detections)} objects")
for det in detections[:5]:  # Show first 5 detections
    print(f"  {det['class_name']}: {det['confidence']:.2f} at {det['center']}")
```

### Object Tracking System

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import cv2

@dataclass
class TrackedObject:
    """
    Represents a tracked object with history and state
    """
    id: int
    class_name: str
    bbox: List[float]  # x1, y1, x2, y2
    center: List[float]  # x, y
    area: float
    confidence: float
    age: int
    last_seen: int
    velocity: List[float]  # vx, vy
    history: List[List[float]]  # List of past centers
    disappeared: int  # Number of frames since last detection

class ObjectTracker:
    """
    Multi-object tracker for humanoid robots
    """
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50):
        """
        Initialize object tracker
        """
        self.next_object_id = 0
        self.objects = {}  # Dictionary of tracked objects
        self.disappeared = {}  # Count of frames since object disappeared
        
        # Parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # For velocity estimation
        self.history_length = 5  # Number of past positions to consider for velocity
        
    def register(self, detection: Dict):
        """
        Register a new object
        """
        new_obj = TrackedObject(
            id=self.next_object_id,
            class_name=detection['class_name'],
            bbox=detection['bbox'],
            center=detection['center'],
            area=detection['area'],
            confidence=detection['confidence'],
            age=0,
            last_seen=0,
            velocity=[0.0, 0.0],
            history=[detection['center']],
            disappeared=0
        )
        
        self.objects[self.next_object_id] = new_obj
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        
        return new_obj.id
    
    def deregister(self, object_id: int):
        """
        Remove an object from tracking
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, detections: List[Dict]) -> Dict[int, TrackedObject]:
        """
        Update tracking with new detections
        """
        # If no detections, increment disappeared counter for all objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Update ages
            for obj in self.objects.values():
                obj.age += 1
                obj.last_seen += 1
            
            return self.objects
        
        # Initialize input centroids for current frame
        input_centroids = np.zeros((len(detections), 2), dtype="float")
        input_bboxes = []
        input_classes = []
        input_confidences = []
        
        for i, detection in enumerate(detections):
            input_centroids[i] = detection['center']
            input_bboxes.append(detection['bbox'])
            input_classes.append(detection['class_name'])
            input_confidences.append(detection['confidence'])
        
        # If no objects are being tracked, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register({
                    'class_name': input_classes[i],
                    'bbox': input_bboxes[i],
                    'center': input_centroids[i].tolist(),
                    'area': (input_bboxes[i][2] - input_bboxes[i][0]) * (input_bboxes[i][3] - input_bboxes[i][1]),
                    'confidence': input_confidences[i]
                })
        else:
            # Match existing objects to new detections
            object_centroids = np.array([obj.center for obj in self.objects.values()])
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix between existing objects and new detections
            D = np.linalg.norm(
                np.repeat(object_centroids[:, np.newaxis, :], len(input_centroids), axis=1) -
                np.repeat(input_centroids[np.newaxis, :, :], len(object_centroids), axis=0),
                axis=2
            )
            
            # Find the minimum values and their indices
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Loop over matched pairs
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                # Get the object ID for the current row, update its information
                object_id = object_ids[row]
                self.objects[object_id].center = input_centroids[col].tolist()
                self.objects[object_id].bbox = input_bboxes[col]
                self.objects[object_id].class_name = input_classes[col]
                self.objects[object_id].confidence = input_confidences[col]
                self.objects[object_id].area = (input_bboxes[col][2] - input_bboxes[col][0]) * (input_bboxes[col][3] - input_bboxes[col][1])
                
                # Update velocity (simplified)
                if len(self.objects[object_id].history) > 0:
                    prev_center = self.objects[object_id].history[-1]
                    dt = 1.0  # Assuming 1 frame interval
                    vx = (self.objects[object_id].center[0] - prev_center[0]) / dt
                    vy = (self.objects[object_id].center[1] - prev_center[1]) / dt
                    self.objects[object_id].velocity = [vx, vy]
                
                # Update history
                self.objects[object_id].history.append(self.objects[object_id].center)
                if len(self.objects[object_id].history) > self.history_length:
                    self.objects[object_id].history.pop(0)
                
                # Reset disappeared counter
                self.disappeared[object_id] = 0
                
                # Update counters
                self.objects[object_id].age += 1
                self.objects[object_id].last_seen = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If some existing objects didn't match any detection, increment disappeared counter
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections that didn't match any existing object
            for col in unused_col_indices:
                new_obj_id = self.register({
                    'class_name': input_classes[col],
                    'bbox': input_bboxes[col],
                    'center': input_centroids[col].tolist(),
                    'area': (input_bboxes[col][2] - input_bboxes[col][0]) * (input_bboxes[col][3] - input_bboxes[col][1]),
                    'confidence': input_confidences[col]
                })
                
                # Update the newly registered object's age and last seen
                self.objects[new_obj_id].age += 1
                self.objects[new_obj_id].last_seen = 0
        
        return self.objects
    
    def predict_next_position(self, object_id: int) -> List[float]:
        """
        Predict next position of an object based on its velocity
        """
        if object_id not in self.objects:
            return None
        
        obj = self.objects[object_id]
        
        # Simple constant velocity prediction
        dt = 1.0  # Predict 1 frame ahead
        next_x = obj.center[0] + obj.velocity[0] * dt
        next_y = obj.center[1] + obj.velocity[1] * dt
        
        return [next_x, next_y]
    
    def get_object_trajectory(self, object_id: int) -> List[List[float]]:
        """
        Get the trajectory of a tracked object
        """
        if object_id not in self.objects:
            return []
        
        return self.objects[object_id].history.copy()
    
    def get_moving_objects(self, velocity_threshold: float = 5.0) -> List[int]:
        """
        Get IDs of objects moving faster than threshold
        """
        moving_ids = []
        for obj_id, obj in self.objects.items():
            speed = np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            if speed > velocity_threshold:
                moving_ids.append(obj_id)
        
        return moving_ids

# Example usage
tracker = ObjectTracker()

# Simulate a sequence of detections
detections_seq = [
    [  # Frame 1
        {'class_name': 'person', 'bbox': [100, 100, 200, 300], 'center': [150, 200], 'area': 10000, 'confidence': 0.9},
        {'class_name': 'chair', 'bbox': [300, 200, 400, 350], 'center': [350, 275], 'area': 15000, 'confidence': 0.8}
    ],
    [  # Frame 2 - person moved, chair stayed the same
        {'class_name': 'person', 'bbox': [105, 105, 205, 305], 'center': [155, 205], 'area': 10000, 'confidence': 0.9},
        {'class_name': 'chair', 'bbox': [300, 200, 400, 350], 'center': [350, 275], 'area': 15000, 'confidence': 0.8}
    ]
]

for frame_idx, frame_dets in enumerate(detections_seq):
    tracked_objects = tracker.update(frame_dets)
    print(f"Frame {frame_idx + 1}: Tracking {len(tracked_objects)} objects")
    
    for obj_id, obj in tracked_objects.items():
        print(f"  Object {obj_id} ({obj.class_name}): center={obj.center}, velocity={obj.velocity}")

# Get moving objects
moving_objects = tracker.get_moving_objects(velocity_threshold=2.0)
print(f"Moving objects: {moving_objects}")
```

## SLAM for humanoid navigation

Simultaneous Localization and Mapping (SLAM) is essential for humanoid robots to navigate unknown environments.

### Visual-Inertial SLAM

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

class VisualInertialSLAM:
    """
    Visual-Inertial SLAM system for humanoid robots
    """
    def __init__(self):
        # State variables
        self.current_pose = np.eye(4)  # Robot's current pose in world frame
        self.pose_graph = []  # List of poses with timestamps
        self.keyframes = []   # Keyframes for mapping
        self.map_points = []  # 3D map points
        
        # IMU integration
        self.imu_bias = np.zeros(6)  # [acc_bias_x, acc_bias_y, acc_bias_z, gyro_bias_x, gyro_bias_y, gyro_bias_z]
        self.imu_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': R.from_quat([0, 0, 0, 1]),  # Identity rotation
            'acceleration': np.zeros(3),
            'angular_velocity': np.zeros(3)
        }
        
        # Visual feature tracking
        self.feature_tracker = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # SLAM parameters
        self.keyframe_threshold = 0.1  # Translation threshold for keyframe selection
        self.rotation_threshold = 0.1  # Rotation threshold for keyframe selection
        self.max_features = 1000
        self.bundle_adjustment_window = 10  # Number of frames for local BA
        
        # Covariance matrices for uncertainty tracking
        self.process_noise_cov = np.eye(15) * 0.01  # Process noise covariance
        self.measurement_noise_cov = np.eye(6) * 0.1  # Measurement noise covariance
        
        # Previous frame data
        self.prev_frame = None
        self.prev_features = None
        self.prev_descriptors = None
        
    def process_frame(self, image: np.ndarray, imu_data: Dict, timestamp: float) -> Dict:
        """
        Process a single frame with associated IMU data
        """
        # Extract features from current image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.feature_tracker.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 50:
            # Insufficient features, return previous state
            return {
                'pose': self.current_pose.copy(),
                'map_points': self.map_points.copy(),
                'status': 'insufficient_features'
            }
        
        result = {
            'pose': self.current_pose.copy(),
            'map_points': self.map_points.copy(),
            'status': 'ok'
        }
        
        # If this is the first frame, initialize
        if self.prev_frame is None:
            self.initialize_first_frame(image, keypoints, descriptors, timestamp)
            self.prev_frame = gray.copy()
            self.prev_features = keypoints.copy()
            self.prev_descriptors = descriptors.copy()
            return result
        
        # Track features between frames
        matches = self.match_features(self.prev_descriptors, descriptors)
        
        if len(matches) < 20:
            # Insufficient matches, use IMU integration
            self.integrate_imu_only(imu_data, timestamp)
            self.prev_frame = gray.copy()
            self.prev_features = keypoints.copy()
            self.prev_descriptors = descriptors.copy()
            result['status'] = 'imu_only'
            return result
        
        # Extract matched points
        prev_pts = np.float32([self.prev_features[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate motion using Essential matrix
        E, mask = cv2.findEssentialMat(
            prev_pts, curr_pts,
            cameraMatrix=self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is not None:
            # Recover pose from Essential matrix
            _, R_rel, t_rel, _ = cv2.recoverPose(E, prev_pts, curr_pts, cameraMatrix=self.camera_matrix)
            
            # Convert to transformation matrix
            T_rel = np.eye(4)
            T_rel[:3, :3] = R_rel
            T_rel[:3, 3] = t_rel.flatten()
            
            # Update current pose
            self.current_pose = self.current_pose @ T_rel
            
            # Check if this frame should be a keyframe
            translation_norm = np.linalg.norm(T_rel[:3, 3])
            rotation_angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            
            is_keyframe = (translation_norm > self.keyframe_translation_threshold or 
                          rotation_angle > self.keyframe_rotation_threshold)
            
            if is_keyframe:
                self.add_keyframe(image, keypoints, descriptors, self.current_pose, timestamp)
        
        # Update previous frame data
        self.prev_frame = gray.copy()
        self.prev_features = keypoints.copy()
        self.prev_descriptors = descriptors.copy()
        
        # Integrate IMU data for more accurate motion estimation
        self.integrate_imu_data(imu_data, timestamp)
        
        return result
    
    def initialize_first_frame(self, image, keypoints, descriptors, timestamp):
        """
        Initialize SLAM with first frame
        """
        # Initialize camera matrix (this would come from calibration)
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ])
        
        # Initialize first pose as identity
        self.current_pose = np.eye(4)
        
        # Add first keyframe
        first_keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'timestamp': timestamp,
            'features_3d': []  # Will be populated as features are triangulated
        }
        
        self.keyframes.append(first_keyframe)
        self.pose_graph.append({
            'pose': self.current_pose.copy(),
            'timestamp': timestamp
        })
    
    def match_features(self, desc1, desc2):
        """
        Match features between two descriptor sets
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = self.matcher.match(desc1, desc2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Keep only the best matches
        return matches[:min(len(matches), 100)]
    
    def add_keyframe(self, image, keypoints, descriptors, pose, timestamp):
        """
        Add a new keyframe to the map
        """
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': pose.copy(),
            'timestamp': timestamp,
            'features_3d': []
        }
        
        self.keyframes.append(keyframe)
        
        # Perform triangulation of features with previous keyframes to build 3D map
        self.triangulate_features_with_previous_keyframes(keyframe)
        
        # Perform local bundle adjustment
        self.perform_local_bundle_adjustment()
    
    def triangulate_features_with_previous_keyframes(self, current_keyframe):
        """
        Triangulate 3D points using current keyframe and previous keyframes
        """
        for i in range(max(0, len(self.keyframes) - 3), len(self.keyframes) - 1):
            prev_keyframe = self.keyframes[i]
            
            # Match features between current and previous keyframe
            matches = self.match_features(prev_keyframe['descriptors'], current_keyframe['descriptors'])
            
            if len(matches) < 10:
                continue
            
            # Get matched points
            prev_pts = np.float32([prev_keyframe['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1, 2)
            curr_pts = np.float32([current_keyframe['keypoints'][m.trainIdx].pt for m in matches]).reshape(-1, 2)
            
            # Get camera poses
            prev_pose = prev_keyframe['pose']
            curr_pose = current_keyframe['pose']
            
            # Get projection matrices
            prev_proj = self.camera_matrix @ np.hstack([prev_pose[:3, :3], prev_pose[:3, 3:4]])
            curr_proj = self.camera_matrix @ np.hstack([curr_pose[:3, :3], curr_pose[:3, 3:4]])
            
            # Triangulate points
            points_4d = cv2.triangulatePoints(prev_proj, curr_proj, prev_pts.T, curr_pts.T)
            
            # Convert to 3D
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Add valid points to map
            for j, (pt_3d, pt_idx_curr, pt_idx_prev) in enumerate(zip(points_3d, 
                                                                      [m.trainIdx for m in matches],
                                                                      [m.queryIdx for m in matches])):
                # Check if point is in front of both cameras
                if pt_3d[2] > 0 and (curr_pose @ np.append(pt_3d, 1))[2] > 0:
                    # Add to map points
                    self.map_points.append({
                        'coordinates': pt_3d,
                        'observations': [
                            {'keyframe_idx': i, 'feature_idx': pt_idx_prev},
                            {'keyframe_idx': len(self.keyframes) - 1, 'feature_idx': pt_idx_curr}
                        ]
                    })
    
    def perform_local_bundle_adjustment(self):
        """
        Perform local bundle adjustment on recent keyframes
        """
        # For simplicity, this is a placeholder implementation
        # In practice, you would use a full bundle adjustment solver
        # like Ceres Solver or g2o
        
        # Just optimize the last few keyframes and connecting 3D points
        window_size = min(self.bundle_adjustment_window, len(self.keyframes))
        
        if len(self.keyframes) < 2:
            return
        
        # This is a simplified approach - in reality, bundle adjustment
        # involves optimizing both camera poses and 3D point positions
        # to minimize reprojection errors
        
        # For now, just update keyframe poses based on optimized relative motions
        pass
    
    def integrate_imu_data(self, imu_data, timestamp):
        """
        Integrate IMU data to refine pose estimation
        """
        # Extract IMU measurements
        acc_measurement = np.array(imu_data['linear_acceleration'])
        gyro_measurement = np.array(imu_data['angular_velocity'])
        
        # Apply bias correction
        acc_corrected = acc_measurement - self.imu_bias[:3]
        gyro_corrected = gyro_measurement - self.imu_bias[3:]
        
        # Integrate accelerometer data to get velocity and position
        dt = timestamp - self.last_imu_timestamp if hasattr(self, 'last_imu_timestamp') else 0.01
        self.last_imu_timestamp = timestamp
        
        if dt <= 0:
            dt = 0.01  # Default to 10ms if no previous timestamp
        
        # Update velocity and position using accelerometer data
        current_orientation = R.from_matrix(self.current_pose[:3, :3])
        acc_world_frame = current_orientation.apply(acc_corrected)
        
        self.imu_state['velocity'] += acc_world_frame * dt
        self.imu_state['position'] += self.imu_state['velocity'] * dt
        
        # Update orientation using gyroscope data
        angular_displacement = gyro_corrected * dt
        delta_rotation = R.from_rotvec(angular_displacement)
        self.imu_state['orientation'] = self.imu_state['orientation'] * delta_rotation
        
        # Update robot pose based on IMU integration
        imu_pose = np.eye(4)
        imu_pose[:3, :3] = self.imu_state['orientation'].as_matrix()
        imu_pose[:3, 3] = self.imu_state['position']
        
        # Fuse visual and IMU estimates (simplified approach)
        # In practice, you would use a more sophisticated fusion method like EKF or UKF
        fusion_factor = 0.7  # Give more weight to visual odometry
        self.current_pose[:3, :3] = (
            fusion_factor * self.current_pose[:3, :3] + 
            (1 - fusion_factor) * imu_pose[:3, :3]
        )
        self.current_pose[:3, 3] = (
            fusion_factor * self.current_pose[:3, 3] + 
            (1 - fusion_factor) * imu_pose[:3, 3]
        )
    
    def get_current_map(self):
        """
        Get the current map as a point cloud
        """
        if not self.map_points:
            return o3d.geometry.PointCloud()
        
        points = np.array([point['coordinates'] for point in self.map_points])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def get_trajectory(self):
        """
        Get the robot's trajectory
        """
        if not self.pose_graph:
            return np.array([])
        
        positions = np.array([pose['pose'][:3, 3] for pose in self.pose_graph])
        return positions

# Example usage
vislam = VisualInertialSLAM()

# Simulate processing a sequence of frames with IMU data
for i in range(10):
    # Simulate image and IMU data
    simulated_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    simulated_imu = {
        'linear_acceleration': [0.1, 0.0, 9.8],  # Simulated gravity + small acceleration
        'angular_velocity': [0.01, 0.02, 0.03]   # Small angular velocities
    }
    
    slam_result = vislam.process_frame(simulated_image, simulated_imu, i * 0.1)
    
    print(f"Frame {i+1}: Pose updated, {len(vislam.map_points)} map points, {len(vislam.keyframes)} keyframes")
```

### Mapping and Localization

```python
class MapManager:
    """
    Manages the map and localization for humanoid robots
    """
    def __init__(self, resolution=0.05):
        self.resolution = resolution  # meters per cell
        self.map_origin = np.array([0.0, 0.0, 0.0])
        self.occupancy_grid = np.zeros((100, 100), dtype=np.int8)  # Initially empty
        self.probability_map = np.full((100, 100), 0.5, dtype=np.float32)  # Unknown areas
        self.visited_cells = set()  # Track which cells have been visited
        
        # Global localization
        self.particle_filter = None
        self.initialized = False
        
        # Map management parameters
        self.max_map_size = 200  # Maximum cells in each dimension
        self.free_threshold = 0.2  # Probability below which cell is considered free
        self.occupied_threshold = 0.65  # Probability above which cell is considered occupied
        
    def update_map_with_sensor_data(self, sensor_data, robot_pose):
        """
        Update the map with new sensor data
        """
        # Transform sensor data to map coordinates
        sensor_points = self.transform_sensor_data_to_map(sensor_data, robot_pose)
        
        # Update occupancy probabilities
        for point in sensor_points:
            map_x, map_y = self.world_to_map_coordinates(point[:2])
            
            if 0 <= map_x < self.occupancy_grid.shape[1] and 0 <= map_y < self.occupancy_grid.shape[0]:
                # Update using probabilistic occupancy grid mapping
                current_prob = self.probability_map[map_y, map_x]
                
                # Use inverse sensor model to update probabilities
                if point[2] < 0.1:  # Likely an obstacle (close measurement)
                    # Apply occupancy update
                    odds = current_prob / (1 - current_prob + 1e-6)
                    new_odds = odds * 3.0  # Increase odds of occupancy
                    new_prob = new_odds / (1 + new_odds)
                else:  # Free space measurement
                    # Apply free space update
                    odds = current_prob / (1 - current_prob + 1e-6)
                    new_odds = odds * 0.3  # Decrease odds of occupancy
                    new_prob = new_odds / (1 + new_odds)
                
                self.probability_map[map_y, map_x] = new_prob
                self.visited_cells.add((map_x, map_y))
        
        # Update occupancy grid based on probabilities
        self.occupancy_grid = np.where(
            self.probability_map > self.occupied_threshold, 
            100,  # Occupied
            np.where(
                self.probability_map < self.free_threshold,
                0,   # Free
                -1   # Unknown
            )
        )
    
    def transform_sensor_data_to_map(self, sensor_data, robot_pose):
        """
        Transform sensor measurements to map coordinates
        """
        # This would transform sensor data (e.g., from LIDAR, depth camera) 
        # to map coordinates based on robot pose
        # For now, we'll simulate this with a simple transformation
        
        # Get rotation and translation from robot pose
        rotation = robot_pose[:3, :3]
        translation = robot_pose[:3, 3]
        
        # Simulated sensor points in robot frame
        sensor_points_robot_frame = np.array(sensor_data['points'])  # This would come from actual sensor
        
        # Transform to world frame
        sensor_points_world = (rotation @ sensor_points_robot_frame.T).T + translation
        
        return sensor_points_world
    
    def world_to_map_coordinates(self, world_coords):
        """
        Convert world coordinates to map grid coordinates
        """
        world_x, world_y = world_coords
        map_x = int((world_x - self.map_origin[0]) / self.resolution)
        map_y = int((world_y - self.map_origin[1]) / self.resolution)
        return map_x, map_y
    
    def map_to_world_coordinates(self, map_coords):
        """
        Convert map grid coordinates to world coordinates
        """
        map_x, map_y = map_coords
        world_x = map_x * self.resolution + self.map_origin[0]
        world_y = map_y * self.resolution + self.map_origin[1]
        return world_x, world_y
    
    def expand_map_if_needed(self, robot_position):
        """
        Expand the map if the robot is approaching the boundary
        """
        world_x, world_y = robot_position[:2]
        map_x, map_y = self.world_to_map_coordinates([world_x, world_y])
        
        # Check if we need to expand the map
        margin = 20  # Cells of margin
        if (map_x < margin or map_x >= self.occupancy_grid.shape[1] - margin or
            map_y < margin or map_y >= self.occupancy_grid.shape[0] - margin):
            
            # Calculate new map size and origin
            new_width = max(self.occupancy_grid.shape[1] * 2, 200)
            new_height = max(self.occupancy_grid.shape[0] * 2, 200)
            
            # Calculate new origin to center the robot
            new_origin_x = world_x - (new_width / 2) * self.resolution
            new_origin_y = world_y - (new_height / 2) * self.resolution
            
            # Create new larger maps
            new_probability_map = np.full((new_height, new_width), 0.5, dtype=np.float32)
            new_occupancy_grid = np.zeros((new_height, new_width), dtype=np.int8)
            
            # Copy old map to center of new map
            old_h, old_w = self.probability_map.shape
            start_y = (new_height - old_h) // 2
            start_x = (new_width - old_w) // 2
            
            new_probability_map[start_y:start_y+old_h, start_x:start_x+old_w] = self.probability_map
            new_occupancy_grid[start_y:start_y+old_h, start_x:start_x+old_w] = self.occupancy_grid
            
            # Update map parameters
            self.probability_map = new_probability_map
            self.occupancy_grid = new_occupancy_grid
            self.map_origin = np.array([new_origin_x, new_origin_y, 0.0])
    
    def find_path(self, start, goal, path_type='a_star'):
        """
        Find a path from start to goal in the map
        """
        if path_type == 'a_star':
            return self.a_star_path(start, goal)
        elif path_type == 'dijkstra':
            return self.dijkstra_path(start, goal)
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def a_star_path(self, start, goal):
        """
        A* pathfinding implementation
        """
        import heapq
        
        start_map = self.world_to_map_coordinates(start[:2])
        goal_map = self.world_to_map_coordinates(goal[:2])
        
        # Check if start and goal are within map bounds
        if (not (0 <= start_map[0] < self.occupancy_grid.shape[1] and 
                 0 <= start_map[1] < self.occupancy_grid.shape[0]) or
            not (0 <= goal_map[0] < self.occupancy_grid.shape[1] and 
                 0 <= goal_map[1] < self.occupancy_grid.shape[0])):
            return []  # Invalid coordinates
        
        # Check if start or goal are in occupied cells
        if self.occupancy_grid[start_map[1], start_map[0]] == 100 or \
           self.occupancy_grid[goal_map[1], goal_map[0]] == 100:
            return []  # Start or goal in obstacle
        
        # A* algorithm
        heap = [(0, start_map)]
        came_from = {}
        cost_so_far = {start_map: 0}
        came_from[start_map] = None
        
        while heap:
            current_cost, current = heapq.heappop(heap)
            
            if current == goal_map:
                break
            
            # Check 8-connected neighborhood
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip current cell
                    
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check bounds
                    if (0 <= neighbor[0] < self.occupancy_grid.shape[1] and 
                        0 <= neighbor[1] < self.occupancy_grid.shape[0]):
                        
                        # Skip if occupied
                        if self.occupancy_grid[neighbor[1], neighbor[0]] == 100:
                            continue
                        
                        # Calculate movement cost (diagonal movement costs more)
                        move_cost = np.sqrt(2) if dx != 0 and dy != 0 else 1.0
                        new_cost = cost_so_far[current] + move_cost
                        
                        if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                            cost_so_far[neighbor] = new_cost
                            priority = new_cost + self.heuristic(neighbor, goal_map)
                            heapq.heappush(heap, (priority, neighbor))
                            came_from[neighbor] = current
        
        # Reconstruct path
        if goal_map not in came_from:
            return []  # No path found
        
        path = []
        current = goal_map
        while current != start_map:
            path.append(current)
            current = came_from[current]
        path.append(start_map)
        path.reverse()
        
        # Convert back to world coordinates
        world_path = []
        for map_coord in path:
            world_x, world_y = self.map_to_world_coordinates(map_coord)
            world_path.append([world_x, world_y, start[2]])  # Keep original Z coordinate
        
        return world_path
    
    def heuristic(self, a, b):
        """
        Heuristic function for A* (Manhattan distance with diagonal movement consideration)
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return min(dx, dy) * np.sqrt(2) + abs(dx - dy)
    
    def get_traversable_area(self, robot_radius):
        """
        Get the area that is traversable for a robot with given radius
        """
        from scipy.ndimage import binary_erosion
        
        # Convert occupancy grid to binary (1 for free space, 0 for occupied)
        binary_map = (self.occupancy_grid == 0).astype(np.uint8)
        
        # Calculate erosion kernel size based on robot radius
        kernel_size = int(robot_radius / self.resolution) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Erode the map to account for robot size
        traversable_map = binary_erosion(binary_map, structure=kernel).astype(np.uint8)
        
        return traversable_map
    
    def localize_robot(self, sensor_data, robot_pose_guess):
        """
        Perform localization using sensor data and map
        """
        # This would implement a particle filter or other localization algorithm
        # For now, we'll return the guess with a simple validation
        if self.is_pose_valid(robot_pose_guess):
            return robot_pose_guess
        else:
            # Try to find a valid nearby pose
            return self.find_valid_nearby_pose(robot_pose_guess)
    
    def is_pose_valid(self, pose):
        """
        Check if a pose is valid (not in an occupied cell)
        """
        map_x, map_y = self.world_to_map_coordinates(pose[:2])
        
        if (0 <= map_x < self.occupancy_grid.shape[1] and 
            0 <= map_y < self.occupancy_grid.shape[0]):
            return self.occupancy_grid[map_y, map_x] != 100  # Not occupied
        else:
            return False  # Outside map
    
    def find_valid_nearby_pose(self, pose_guess, max_attempts=10):
        """
        Find a valid nearby pose if the current pose is invalid
        """
        for attempt in range(max_attempts):
            # Try random offsets
            offset_x = np.random.uniform(-0.5, 0.5)
            offset_y = np.random.uniform(-0.5, 0.5)
            
            candidate_pose = pose_guess.copy()
            candidate_pose[0] += offset_x
            candidate_pose[1] += offset_y
            
            if self.is_pose_valid(candidate_pose):
                return candidate_pose
        
        # If no valid nearby pose found, return original guess
        return pose_guess

# Example usage
map_manager = MapManager(resolution=0.1)

# Example of updating map with sensor data
example_sensor_data = {
    'points': [[0.5, 0.0, 0.0], [0.6, 0.1, 0.0], [0.7, -0.1, 0.0]]  # Simulated sensor points
}

# Simulated robot pose
robot_pose = np.eye(4)
robot_pose[0, 3] = 1.0  # x position
robot_pose[1, 3] = 0.0  # y position
robot_pose[2, 3] = 0.8  # z position

# Update map with sensor data
map_manager.update_map_with_sensor_data(example_sensor_data, robot_pose)

# Find a path
start_pos = [0.5, 0.0, 0.8]
goal_pos = [2.0, 1.0, 0.8]
path = map_manager.a_star_path(start_pos, goal_pos)

print(f"Found path with {len(path)} waypoints")
if path:
    print(f"Path starts at {path[0]} and ends at {path[-1]}")
else:
    print("No path found")