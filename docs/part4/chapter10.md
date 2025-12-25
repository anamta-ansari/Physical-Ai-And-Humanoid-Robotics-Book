---
title: Isaac ROS
sidebar_position: 4
description: Isaac ROS integration with ROS 2, hardware-accelerated perception, and GPU-accelerated robotics applications
---

# Isaac ROS

## Isaac ROS integration with ROS 2

Isaac ROS is NVIDIA's collection of hardware-accelerated perception and navigation packages designed specifically for robotics applications. It bridges the gap between NVIDIA's GPU computing capabilities and the ROS 2 ecosystem, enabling robots to perform complex perception tasks in real-time.

### Introduction to Isaac ROS

Isaac ROS provides a collection of GPU-accelerated packages that implement common robotics perception and navigation tasks:

1. **Hardware Acceleration**: Leverages CUDA, TensorRT, and other NVIDIA technologies
2. **ROS 2 Native**: Follows ROS 2 conventions and interfaces
3. **Modular Design**: Can be used individually or combined
4. **Performance Optimized**: Significantly faster than CPU-only implementations
5. **Industrial Quality**: Designed for production robotics applications

### Core Isaac ROS Packages

1. **Isaac ROS Image Pipeline**: Accelerated image processing and rectification
2. **Isaac ROS AprilTag Detection**: High-speed fiducial marker detection
3. **Isaac ROS Stereo Dense Reconstruction**: Depth estimation from stereo cameras
4. **Isaac ROS Visual Slam**: Visual SLAM with GPU acceleration
5. **Isaac ROS DetectNet**: Object detection with TensorRT acceleration
6. **Isaac ROS Segmentation**: Semantic segmentation using GPU acceleration
7. **Isaac ROS Point Cloud Utilities**: Accelerated point cloud processing
8. **Isaac ROS Occupancy Grids**: GPU-accelerated occupancy grid generation

### Installation and Setup

```bash
# Install Isaac ROS dependencies
sudo apt update
sudo apt install -y ros-humble-isaac-ros-common
sudo apt install -y ros-humble-isaac-ros-perception
sudo apt install -y ros-humble-isaac-ros-navigation
```

### Basic Integration Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch

class IsaacROSIntegrationNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_integration_node')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Create subscribers for camera and depth data
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )
        
        # Create publishers for processed data
        self.detection_pub = self.create_publisher(Detection2DArray, '/isaac_ros/detections', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/isaac_ros/pointcloud', 10)
        
        # Isaac ROS specific parameters
        self.declare_parameter('use_tensor_rt', True)
        self.declare_parameter('tensor_rt_precision', 'FP16')
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('max_objects', 10)
        
        self.use_tensor_rt = self.get_parameter('use_tensor_rt').value
        self.tensor_rt_precision = self.get_parameter('tensor_rt_precision').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.max_objects = self.get_parameter('max_objects').value
        
        # Initialize Isaac ROS perception nodes (these would be actual Isaac ROS nodes)
        self.initialize_isaac_perception_nodes()
        
        self.get_logger().info('Isaac ROS Integration Node initialized')
    
    def initialize_isaac_perception_nodes(self):
        """
        Initialize Isaac ROS perception nodes
        In practice, these would be launched separately via launch files
        """
        # Placeholder for Isaac ROS node initialization
        # In a real implementation, this would connect to actual Isaac ROS packages:
        # - Isaac ROS Image Pipeline
        # - Isaac ROS Detection Pipeline  
        # - Isaac ROS Depth Pipeline
        # - Isaac ROS PointCloud Pipeline
        
        # For this example, we'll simulate the functionality
        self.isaac_initialized = True
        
        # Load Isaac ROS compatible models
        try:
            # Example: Load Isaac ROS detection model (placeholder)
            self.detection_model = self.load_isaac_detection_model()
            self.get_logger().info('Isaac ROS detection model loaded')
        except ImportError:
            self.get_logger().warn('Isaac ROS packages not available, using CPU fallback')
            self.detection_model = None
    
    def load_isaac_detection_model(self):
        """
        Load Isaac ROS compatible detection model
        """
        # In a real implementation, this would load an Isaac ROS detection model
        # For example, using Isaac ROS DetectNet or other perception models
        try:
            # Placeholder - in reality this would load a TensorRT optimized model
            import torch
            # Model loading would happen here
            return torch.nn.Identity()  # Placeholder model
        except Exception as e:
            self.get_logger().error(f'Failed to load Isaac ROS model: {str(e)}')
            return None
    
    def image_callback(self, msg):
        """
        Process image using Isaac ROS pipeline
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Apply Isaac ROS perception pipeline
            if self.use_tensor_rt and self.detection_model:
                detections = self.process_with_isaac_detection(cv_image)
            else:
                # Fallback to CPU processing
                detections = self.process_with_cpu_detection(cv_image)
            
            # Publish results
            if detections:
                detection_msg = Detection2DArray()
                detection_msg.header = msg.header
                detection_msg.detections = detections
                self.detection_publisher.publish(detection_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS pipeline: {str(e)}')
    
    def depth_callback(self, msg):
        """
        Process depth image using Isaac ROS pipeline
        """
        try:
            # Convert ROS depth image to OpenCV format
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # Apply Isaac ROS depth processing
            pointcloud = self.process_with_isaac_depth(cv_depth, msg.header)
            
            # Publish results
            if pointcloud:
                self.pointcloud_publisher.publish(pointcloud)
                
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS depth processing: {str(e)}')
    
    def process_with_isaac_detection(self, image):
        """
        Process image with Isaac ROS detection pipeline
        """
        # This would use Isaac ROS's GPU-accelerated detection
        # For this example, we'll simulate the functionality
        try:
            import torch
            
            # Preprocess image for Isaac ROS model
            input_tensor = self.preprocess_for_isaac_model(image)
            
            # Run inference using Isaac ROS model (with TensorRT acceleration if enabled)
            with torch.no_grad():
                detections = self.detection_model(input_tensor)
            
            # Post-process detections
            processed_detections = self.post_process_isaac_detections(detections, image.shape)
            
            return processed_detections
            
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS detection: {str(e)}')
            return []
    
    def preprocess_for_isaac_model(self, image):
        """
        Preprocess image for Isaac ROS model input
        """
        # Resize image to model input size (typically 224x224 or 416x416)
        input_size = (416, 416)  # Common size for Isaac ROS models
        resized_image = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and change to CHW format
        normalized_image = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(input_tensor).cuda() if torch.cuda.is_available() else torch.from_numpy(input_tensor)
        
        return input_tensor
    
    def post_process_isaac_detections(self, raw_detections, image_shape):
        """
        Post-process Isaac ROS detection outputs
        """
        # This would convert Isaac ROS model outputs to standard ROS messages
        # In practice, Isaac ROS nodes handle this conversion internally
        
        # Placeholder implementation
        height, width = image_shape[:2]
        
        # Example: create mock detections
        # In reality, this would come from the Isaac ROS model output
        detections = []
        for i in range(min(3, self.max_objects)):  # Example: 3 detections
            detection = Detection2D()
            
            # Random bounding box (in practice, comes from model)
            x = np.random.randint(0, width - 100)
            y = np.random.randint(0, height - 100)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            detection.bbox.center.x = x + w/2
            detection.bbox.center.y = y + h/2
            detection.bbox.size_x = w
            detection.bbox.size_y = h
            
            # Add hypothesis with confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(np.random.choice(['person', 'chair', 'cup', 'bottle']))
            hypothesis.score = float(np.random.uniform(0.6, 0.95))
            detection.results.append(hypothesis)
            
            detections.append(detection)
        
        return detections
    
    def process_with_isaac_depth(self, depth_image, header):
        """
        Process depth image with Isaac ROS pipeline
        """
        try:
            # In practice, Isaac ROS would use hardware-accelerated depth processing
            # This is a simplified example
            
            # Apply depth filtering (in practice, Isaac ROS has optimized filters)
            filtered_depth = self.filter_depth_image(depth_image)
            
            # Generate point cloud from depth
            pointcloud = self.depth_to_pointcloud(filtered_depth, header)
            
            return pointcloud
            
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS depth processing: {str(e)}')
            return None
    
    def filter_depth_image(self, depth_image):
        """
        Apply depth filtering similar to Isaac ROS
        """
        # In Isaac ROS, this would use CUDA-accelerated filtering
        # For this example, we'll use OpenCV filtering
        
        # Apply median filter to reduce noise
        filtered_depth = cv2.medianBlur(depth_image, 5)
        
        # Apply bilateral filter to preserve edges while smoothing
        filtered_depth = cv2.bilateralFilter(filtered_depth, 9, 75, 75)
        
        return filtered_depth
    
    def depth_to_pointcloud(self, depth_image, header):
        """
        Convert depth image to point cloud using camera parameters
        """
        # This would typically get camera parameters from camera_info topic
        # For this example, using default values
        fx = 554.256  # Default focal length (from Kinect)
        fy = 554.256
        cx = 320.5   # Default principal point
        cy = 240.5
        
        height, width = depth_image.shape
        
        # Generate coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert pixel coordinates to camera coordinates
        x_cam = (x_coords - cx) * depth_image / fx
        y_cam = (y_coords - cy) * depth_image / fy
        
        # Stack to get 3D points
        points = np.stack((x_cam, y_cam, depth_image), axis=-1).reshape(-1, 3)
        
        # Remove invalid points (where depth is 0 or invalid)
        valid_mask = (depth_image > 0) & (depth_image < 10.0)  # Valid depth range
        valid_points = points[valid_mask.flatten()]
        
        # Create PointCloud2 message
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_depth_optical_frame'
        
        # Create point cloud
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        pointcloud_msg = point_cloud2.create_cloud(header, fields, valid_points)
        
        return pointcloud_msg

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacROSIntegrationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac ROS Integration Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Launch Configuration

```python
# isaac_ros_perception_pipeline.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    detection_model = LaunchConfiguration('detection_model')
    depth_processing = LaunchConfiguration('depth_processing')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'detection_model',
            default_value='detectnet_coco',
            description='Detection model to use'
        ),
        DeclareLaunchArgument(
            'depth_processing',
            default_value='true',
            description='Enable depth processing pipeline'
        ),
        
        # Isaac ROS Image Pipeline
        Node(
            package='isaac_ros_image_pipeline',
            executable='isaac_ros_image_rect',
            name='image_rect',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[
                ('image_raw', '/camera/rgb/image_raw'),
                ('camera_info', '/camera/rgb/camera_info'),
                ('image_rect', '/camera/rgb/image_rect_color'),
                ('camera_info_rect', '/camera/rgb/camera_info_rect')
            ]
        ),
        
        # Isaac ROS Detection Pipeline
        Node(
            package='isaac_ros_detectnet',
            executable='isaac_ros_detectnet',
            name='detectnet',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'model_name': detection_model},
                {'input_topic': '/camera/rgb/image_rect_color'},
                {'output_topic': '/isaac_ros/detections'},
                {'confidence_threshold': 0.7},
                {'max_objects': 20}
            ],
            remappings=[
                ('image_input', '/camera/rgb/image_rect_color'),
                ('detections_output', '/isaac_ros/detections')
            ]
        ),
        
        # Isaac ROS Depth Pipeline (if enabled)
        Node(
            package='isaac_ros_depth_preprocessor',
            executable='isaac_ros_depth_preprocessor',
            name='depth_preprocessor',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(depth_processing),
            remappings=[
                ('depth_image', '/camera/depth/image_rect_raw'),
                ('camera_info', '/camera/rgb/camera_info'),
                ('processed_depth', '/isaac_ros/processed_depth')
            ]
        ),
        
        # Isaac ROS Point Cloud Pipeline
        Node(
            package='isaac_ros_pointcloud_utils',
            executable='isaac_ros_pointcloud_creator',
            name='pointcloud_creator',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(depth_processing),
            remappings=[
                ('depth_image', '/isaac_ros/processed_depth'),
                ('image', '/camera/rgb/image_rect_color'),
                ('camera_info', '/camera/rgb/camera_info'),
                ('pointcloud', '/isaac_ros/pointcloud')
            ]
        ),
        
        # Isaac ROS Stereo Disparity (for stereo cameras)
        Node(
            package='isaac_ros_stereo_image_proc',
            executable='isaac_ros_disparity_node',
            name='disparity_node',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('use_stereo')),
            remappings=[
                ('left/image_rect', '/camera/left/image_rect'),
                ('right/image_rect', '/camera/right/image_rect'),
                ('left/camera_info', '/camera/left/camera_info'),
                ('right/camera_info', '/camera/right/camera_info'),
                ('disparity', '/camera/disparity')
            ]
        ),
        
        # Isaac ROS Object Stereo (for 3D object detection from stereo)
        Node(
            package='isaac_ros_object_segmentation',
            executable='isaac_ros_object_stereo_node',
            name='object_stereo',
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('use_stereo')),
            remappings=[
                ('disparity', '/camera/disparity'),
                ('image', '/camera/left/image_rect'),
                ('segmented_objects', '/isaac_ros/segmented_3d_objects')
            ]
        ),
        
        LogInfo(
            msg=['Isaac ROS Perception Pipeline launched with model: ', detection_model]
        )
    ])
```

## Hardware-accelerated perception

Isaac ROS provides GPU-accelerated perception capabilities that significantly outperform CPU-based approaches for computationally intensive tasks.

### GPU Acceleration Framework

```python
import numpy as np
import cv2
import torch
import torch.nn as nn
import cupy as cp  # For CUDA operations
from numba import cuda
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from typing import List, Tuple, Dict, Any

class IsaacROSGPUAcceleratedPerception(Node):
    def __init__(self):
        super().__init__('isaac_ros_gpu_perception')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Check for GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device('cuda')
            self.get_logger().info('GPU acceleration enabled')
        else:
            self.device = torch.device('cpu')
            self.get_logger().warn('GPU not available, using CPU for perception')
        
        # Initialize CUDA operations if available
        if self.gpu_available:
            self.initialize_cuda_kernels()
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/isaac_ros/gpu_detections', 10
        )
        
        self.feature_pub = self.create_publisher(
            PointStamped, '/isaac_ros/gpu_features', 10
        )
        
        # Initialize perception models
        self.initialize_models()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        self.get_logger().info('Isaac ROS GPU Perception Pipeline initialized')
    
    def initialize_cuda_kernels(self):
        """
        Initialize custom CUDA kernels for perception operations
        """
        # Define CUDA kernel for image preprocessing
        self.cuda_preprocess_kernel = """
        extern "C" __global__
        void preprocess_image(const unsigned char* input, float* output, 
                             int width, int height, float mean_r, float mean_g, float mean_b,
                             float std_r, float std_g, float std_b) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (idx < width && idy < height) {
                int pixel_idx = idy * width + idx;
                
                // Convert BGR to RGB and normalize
                output[pixel_idx * 3 + 0] = (input[pixel_idx * 3 + 2] / 255.0f - mean_r) / std_r;  // R
                output[pixel_idx * 3 + 1] = (input[pixel_idx * 3 + 1] / 255.0f - mean_g) / std_g;  // G
                output[pixel_idx * 3 + 2] = (input[pixel_idx * 3 + 0] / 255.0f - mean_b) / std_b;  // B
            }
        }
        """
        
        # In practice, you'd compile and load this kernel
        # For this example, we'll use PyTorch operations instead
    
    def initialize_models(self):
        """
        Initialize GPU-accelerated perception models
        """
        try:
            # Initialize object detection model
            self.detection_model = torch.hub.load(
                'ultralytics/yolov5', 'yolov5s', pretrained=True
            ).to(self.device)
            self.detection_model.eval()
            
            # Set model to half precision if using TensorRT-style optimization
            if self.gpu_available:
                self.detection_model.half()  # FP16 for faster inference
            
            # Initialize feature extraction model
            self.feature_model = torch.hub.load(
                'pytorch/vision:v0.10.0', 'resnet50', pretrained=True
            ).to(self.device)
            self.feature_model.eval()
            
            # Initialize depth estimation model
            self.depth_model = torch.hub.load(
                'intel-isl/MiDaS', 'MiDaS_small', pretrained=True
            ).to(self.device)
            self.depth_model.eval()
            
            self.get_logger().info('Perception models initialized on GPU')
            
        except Exception as e:
            self.get_logger().error(f'Error initializing models: {str(e)}')
            raise
    
    def image_callback(self, msg):
        """
        Process incoming image with GPU-accelerated perception
        """
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process with GPU-accelerated pipeline
            start_time = self.get_clock().now().nanoseconds / 1e9
            
            # Run object detection
            detections = self.run_gpu_detection(cv_image)
            
            # Extract features
            features = self.run_gpu_feature_extraction(cv_image)
            
            # Estimate depth (if needed)
            # depth_map = self.run_gpu_depth_estimation(cv_image)
            
            # Calculate processing time
            end_time = self.get_clock().now().nanoseconds / 1e9
            processing_time = (end_time - start_time) * 1000  # milliseconds
            
            # Publish results
            if detections:
                self.publish_detections(detections, msg.header)
            
            if features:
                self.publish_features(features, msg.header)
            
            # Track performance
            self.frame_count += 1
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.frame_count % 100 == 0:
                avg_fps = self.frame_count / (current_time - self.start_time)
                self.get_logger().info(
                    f'Processed {self.frame_count} frames. '
                    f'Avg FPS: {avg_fps:.2f}, Avg processing time: {processing_time:.2f}ms'
                )
                
        except Exception as e:
            self.get_logger().error(f'Error in GPU perception pipeline: {str(e)}')
    
    def run_gpu_detection(self, image):
        """
        Run GPU-accelerated object detection
        """
        try:
            # Preprocess image for model
            input_tensor = self.preprocess_image_for_detection(image)
            
            # Run inference
            with torch.no_grad():
                if self.gpu_available:
                    input_tensor = input_tensor.half()  # Use FP16 if GPU available
                    results = self.detection_model(input_tensor)
                    
                    # Process results
                    detections = self.process_detection_results(results, image.shape)
                    
                    return detections
                else:
                    # Fallback to CPU
                    input_tensor = input_tensor.float()
                    results = self.detection_model(input_tensor)
                    detections = self.process_detection_results(results, image.shape)
                    
                    return detections
        
        except Exception as e:
            self.get_logger().error(f'Error in GPU detection: {str(e)}')
            return None
    
    def preprocess_image_for_detection(self, image):
        """
        Preprocess image for object detection model
        """
        # Resize image to model input size (640x640 for YOLOv5)
        input_size = (640, 640)
        resized_image = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and change to CHW format
        normalized_image = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Convert to torch tensor and move to device
        input_tensor = torch.from_numpy(input_tensor).to(self.device)
        
        return input_tensor
    
    def process_detection_results(self, results, original_shape):
        """
        Process raw detection results into ROS messages
        """
        # Get detections (results.pred[0] contains the detections for batch 0)
        detections = results.pred[0]
        
        if detections is None or len(detections) == 0:
            return None
        
        # Convert to Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_rgb_optical_frame'  # Will be set by caller
        
        height, width = original_shape[:2]
        scale_x = width / 640.0  # Original model input size
        scale_y = height / 640.0
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Create Detection2D message
            det_msg = Detection2D()
            
            # Convert to center coordinates and normalize
            center_x = (x1 + x2) / 2.0 * scale_x
            center_y = (y1 + y2) / 2.0 * scale_y
            size_x = (x2 - x1) * scale_x
            size_y = (y2 - y1) * scale_y
            
            det_msg.bbox.center.x = float(center_x)
            det_msg.bbox.center.y = float(center_y)
            det_msg.bbox.size_x = float(size_x)
            det_msg.bbox.size_y = float(size_y)
            
            # Add classification result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(class_id)
            hypothesis.score = float(confidence)
            det_msg.results.append(hypothesis)
            
            detection_array.detections.append(det_msg)
        
        return detection_array
    
    def run_gpu_feature_extraction(self, image):
        """
        Extract GPU-accelerated features from image
        """
        try:
            # Preprocess image for feature extraction
            input_tensor = self.preprocess_image_for_features(image)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_model(input_tensor)
                
                # Global average pooling to get a fixed-size feature vector
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                feature_vector = pooled_features.view(pooled_features.size(0), -1)
                
                return feature_vector.cpu().numpy()
        
        except Exception as e:
            self.get_logger().error(f'Error in GPU feature extraction: {str(e)}')
            return None
    
    def preprocess_image_for_features(self, image):
        """
        Preprocess image for feature extraction model
        """
        # Resize to model input size (224x224 for ResNet50)
        input_size = (224, 224)
        resized_image = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize using ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        normalized_image = (rgb_image / 255.0 - mean) / std
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Convert to torch tensor and move to device
        input_tensor = torch.from_numpy(input_tensor).to(self.device).float()
        
        return input_tensor
    
    def run_gpu_depth_estimation(self, image):
        """
        Run GPU-accelerated depth estimation
        """
        try:
            # Preprocess image for depth estimation
            input_tensor = self.preprocess_image_for_depth(image)
            
            # Estimate depth
            with torch.no_grad():
                depth_pred = self.depth_model(input_tensor)
                
                # Post-process depth
                depth_map = self.postprocess_depth_prediction(depth_pred)
                
                return depth_map
        
        except Exception as e:
            self.get_logger().error(f'Error in GPU depth estimation: {str(e)}')
            return None
    
    def preprocess_image_for_depth(self, image):
        """
        Preprocess image for depth estimation model
        """
        # Resize to model input size (384x384 for MiDaS small)
        input_size = (384, 384)
        resized_image = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized_image = rgb_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized_image, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Convert to torch tensor and move to device
        input_tensor = torch.from_numpy(input_tensor).to(self.device).float()
        
        return input_tensor
    
    def postprocess_depth_prediction(self, depth_pred):
        """
        Post-process depth prediction
        """
        # Apply activation function and scale
        depth_map = torch.sigmoid(depth_pred)
        
        # Convert to numpy array
        depth_map_np = depth_map.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
        
        return depth_map_np
    
    def publish_detections(self, detections, header):
        """
        Publish detection results
        """
        # Set appropriate header
        detections.header = header
        self.detection_publisher.publish(detections)
    
    def publish_features(self, features, header):
        """
        Publish extracted features
        """
        # For now, just log the feature vector shape
        self.get_logger().debug(f'Extracted features shape: {features.shape}')

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacROSGPUAcceleratedPerception()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac ROS GPU Perception Pipeline')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### TensorRT Optimization for Perception

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import torch.nn.functional as F

class TensorRTOptimizer:
    """
    Utility class for optimizing perception models with TensorRT
    """
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.engine = None
    
    def optimize_model(self, pytorch_model, input_shape, precision='fp16'):
        """
        Optimize a PyTorch model with TensorRT
        """
        # Create builder configuration
        config = self.builder.create_builder_config()
        
        # Set memory limit (in bytes)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # Create optimization profile
        profile = self.builder.create_optimization_profile()
        profile.set_shape(
            "input",  # Input name
            min=(1, *input_shape[1:]),    # Minimum shape
            opt=(4, *input_shape[1:]),    # Optimal shape  
            max=(8, *input_shape[1:]),    # Maximum shape
        )
        config.add_optimization_profile(profile)
        
        # Set precision
        if precision == 'fp16':
            if self.builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("FP16 not supported on this platform, using FP32")
        
        # Convert PyTorch model to ONNX first
        onnx_model_path = "/tmp/temp_model.onnx"
        dummy_input = torch.randn(input_shape).cuda()
        
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Parse ONNX model
        parser = trt.OnnxParser(self.network, self.logger)
        with open(onnx_model_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        serialized_engine = self.builder.build_serialized_network(self.network, config)
        
        if serialized_engine is None:
            print("Engine building failed")
            return None
        
        # Create runtime and engine
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        return engine
    
    def create_tensorrt_model(self, engine):
        """
        Create a callable model from TensorRT engine
        """
        class TRTModel:
            def __init__(self, engine):
                self.engine = engine
                self.context = self.engine.create_execution_context()
                
                # Allocate I/O buffers
                self.input_binding_idx = self.engine.get_binding_index("input")
                self.output_binding_idx = self.engine.get_binding_index("output")
                
                self.max_batch_size = self.engine.max_batch_size
                
                # Get data types and shapes
                input_dtype = self.engine.get_binding_dtype(self.input_binding_idx)
                output_dtype = self.engine.get_binding_dtype(self.output_binding_idx)
                
                input_shape = self.engine.get_binding_shape(self.input_binding_idx)
                output_shape = self.engine.get_binding_shape(self.output_binding_idx)
                
                # Allocate CUDA memory
                self.input_size = trt.volume(input_shape) * self.max_batch_size * np.dtype(np.float32).itemsize
                self.output_size = trt.volume(output_shape) * self.max_batch_size * np.dtype(np.float32).itemsize
                
                self.d_input = cuda.mem_alloc(self.input_size)
                self.d_output = cuda.mem_alloc(self.output_size)
                
                # Create stream
                self.stream = cuda.Stream()
            
            def __call__(self, input_data):
                """
                Run inference on input data
                """
                # Allocate host memory for input/output
                h_input = np.ascontiguousarray(input_data, dtype=np.float32)
                h_output = np.empty(trt.volume(self.engine.get_binding_shape(self.output_binding_idx)) * 
                                   self.max_batch_size, dtype=np.float32)
                
                # Transfer input data to device
                cuda.memcpy_htod_async(self.d_input, h_input, self.stream)
                
                # Run inference
                bindings = [int(self.d_input), int(self.d_output)]
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                
                # Transfer predictions back
                cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
                
                # Synchronize threads
                self.stream.synchronize()
                
                return h_output
        
        return TRTModel(engine)

# Example usage in perception pipeline
class OptimizedIsaacROSPipeline(IsaacROSGPUAcceleratedPerception):
    def __init__(self):
        super().__init__()
        
        # Initialize TensorRT optimizer
        self.trt_optimizer = TensorRTOptimizer()
        
        # Optimize models with TensorRT if available
        if self.gpu_available:
            self.optimize_models_with_tensorrt()
    
    def optimize_models_with_tensorrt(self):
        """
        Optimize perception models with TensorRT
        """
        try:
            # Optimize detection model
            detection_input_shape = (1, 3, 640, 640)  # Batch, Channels, Height, Width
            self.trt_detection_engine = self.trt_optimizer.optimize_model(
                self.detection_model, detection_input_shape, precision='fp16'
            )
            
            if self.trt_detection_engine:
                self.trt_detection_model = self.trt_optimizer.create_tensorrt_model(
                    self.trt_detection_engine
                )
                self.get_logger().info('Detection model optimized with TensorRT')
            else:
                self.get_logger().warn('Failed to optimize detection model with TensorRT')
                self.trt_detection_model = None
            
            # Optimize feature extraction model
            feature_input_shape = (1, 3, 224, 224)
            self.trt_feature_engine = self.trt_optimizer.optimize_model(
                self.feature_model, feature_input_shape, precision='fp16'
            )
            
            if self.trt_feature_engine:
                self.trt_feature_model = self.trt_optimizer.create_tensorrt_model(
                    self.trt_feature_engine
                )
                self.get_logger().info('Feature extraction model optimized with TensorRT')
            else:
                self.get_logger().warn('Failed to optimize feature extraction model with TensorRT')
                self.trt_feature_model = None
        
        except Exception as e:
            self.get_logger().error(f'Error optimizing models with TensorRT: {str(e)}')
    
    def run_gpu_detection(self, image):
        """
        Run GPU-accelerated object detection with TensorRT optimization
        """
        try:
            if self.trt_detection_model and self.gpu_available:
                # Use optimized TensorRT model
                input_tensor = self.preprocess_image_for_detection(image)
                
                # Convert to contiguous array for TensorRT
                input_array = input_tensor.cpu().numpy()
                
                # Run optimized inference
                results = self.trt_detection_model(input_array)
                
                # Process results
                # Note: This is simplified - actual TensorRT output processing would be more complex
                # and would require proper post-processing based on the model architecture
                
                # For now, fall back to PyTorch processing of the results
                with torch.no_grad():
                    # Convert results back to PyTorch tensor format for processing
                    torch_results = torch.from_numpy(results).to(self.device)
                    detections = self.process_detection_results(torch_results, image.shape)
                    
                    return detections
            else:
                # Fall back to standard PyTorch inference
                return super().run_gpu_detection(image)
        
        except Exception as e:
            self.get_logger().error(f'Error in TensorRT detection: {str(e)}')
            # Fall back to standard method
            return super().run_gpu_detection(image)

def main(args=None):
    rclpy.init(args=args)
    
    node = OptimizedIsaacROSPipeline()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Optimized Isaac ROS Perception Pipeline')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor fusion and integration

Isaac ROS provides advanced sensor fusion capabilities to combine data from multiple sensors for more robust perception.

### Multi-Sensor Data Fusion

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32MultiArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

class IsaacROSSensorFusion(Node):
    def __init__(self):
        super().__init__('isaac_ros_sensor_fusion')
        
        # Initialize data storage
        self.camera_data = None
        self.depth_data = None
        self.imu_data = None
        self.lidar_data = None
        self.last_timestamp = None
        
        # Create subscribers for multiple sensor types
        self.camera_sub = Subscriber(self, Image, '/camera/rgb/image_rect_color')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw')
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)
        
        # Synchronize camera and depth data
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_sub, self.depth_sub], 
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.camera_depth_callback)
        
        # Create publishers for fused data
        self.fused_perception_pub = self.create_publisher(Float32MultiArray, '/isaac_ros/fused_perception', 10)
        self.enhanced_pointcloud_pub = self.create_publisher(PointCloud2, '/isaac_ros/enhanced_pointcloud', 10)
        self.fused_pose_pub = self.create_publisher(PoseStamped, '/isaac_ros/fused_pose', 10)
        
        # Sensor fusion parameters
        self.declare_parameter('fusion_confidence_threshold', 0.7)
        self.declare_parameter('max_fusion_delay', 0.1)  # seconds
        self.declare_parameter('use_imu_for_alignment', True)
        self.declare_parameter('use_lidar_for_validation', True)
        
        self.confidence_threshold = self.get_parameter('fusion_confidence_threshold').value
        self.max_fusion_delay = self.get_parameter('max_fusion_delay').value
        self.use_imu_for_alignment = self.get_parameter('use_imu_for_alignment').value
        self.use_lidar_for_validation = self.get_parameter('use_lidar_for_validation').value
        
        # Camera intrinsic parameters (will be populated from camera_info)
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Fusion algorithms
        self.initialize_fusion_algorithms()
        
        self.get_logger().info('Isaac ROS Sensor Fusion Node initialized')
    
    def initialize_fusion_algorithms(self):
        """
        Initialize sensor fusion algorithms
        """
        # Initialize Kalman filter for state estimation
        self.kalman_filter = self.initialize_kalman_filter()
        
        # Initialize particle filter for multi-modal distributions
        self.particle_filter = self.initialize_particle_filter()
        
        # Initialize data association algorithms
        self.data_association = self.initialize_data_association()
    
    def initialize_kalman_filter(self):
        """
        Initialize Kalman filter for sensor fusion
        """
        # For this example, we'll use a simple implementation
        # In practice, you'd use a more sophisticated approach
        return {
            'state': np.zeros(6),  # [x, y, z, vx, vy, vz]
            'covariance': np.eye(6) * 1000,  # Initial uncertainty
            'process_noise': np.eye(6) * 0.1,
            'measurement_noise': np.eye(3) * 0.5  # For position measurements
        }
    
    def initialize_particle_filter(self):
        """
        Initialize particle filter for multi-modal state estimation
        """
        # For this example, we'll define the structure
        # In practice, you'd implement the full particle filter algorithm
        return {
            'particles': np.random.normal(0, 1, (100, 6)),  # 100 particles with 6D state
            'weights': np.ones(100) / 100,  # Equal initial weights
            'state_dim': 6
        }
    
    def initialize_data_association(self):
        """
        Initialize data association algorithms
        """
        return {
            'association_threshold': 0.3,  # Threshold for associating measurements
            'track_management': {}  # Track objects over time
        }
    
    def camera_info_callback(self, msg):
        """
        Store camera intrinsic parameters
        """
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
    
    def imu_callback(self, msg):
        """
        Store IMU data for fusion
        """
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        }
    
    def lidar_callback(self, msg):
        """
        Store LiDAR data for fusion
        """
        self.lidar_data = {
            'ranges': msg.ranges,
            'intensities': msg.intensities,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        }
    
    def camera_depth_callback(self, camera_msg, depth_msg):
        """
        Process synchronized camera and depth data for fusion
        """
        try:
            # Convert ROS images to OpenCV
            camera_cv = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding='bgr8')
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            
            # Perform sensor fusion
            fused_result = self.fuse_camera_depth_imu_lidar(
                camera_cv, depth_cv, 
                self.imu_data, self.lidar_data,
                camera_msg.header.stamp
            )
            
            # Publish fused result
            if fused_result is not None:
                self.publish_fused_data(fused_result, camera_msg.header)
        
        except Exception as e:
            self.get_logger().error(f'Error in camera-depth callback: {str(e)}')
    
    def fuse_camera_depth_imu_lidar(self, camera_image, depth_image, imu_data, lidar_data, timestamp):
        """
        Fuse data from camera, depth, IMU, and LiDAR sensors
        """
        fused_data = {
            'timestamp': timestamp,
            'objects': [],
            'environment_map': None,
            'robot_pose': None,
            'confidence': 0.0
        }
        
        # Step 1: Process camera data for object detection
        camera_objects = self.process_camera_image(camera_image)
        
        # Step 2: Process depth data for 3D positioning
        if depth_image is not None and self.camera_matrix is not None:
            camera_objects = self.add_depth_information(camera_objects, depth_image, self.camera_matrix)
        
        # Step 3: Use IMU data for orientation correction
        if imu_data is not None and self.use_imu_for_alignment:
            camera_objects = self.correct_for_orientation(camera_objects, imu_data)
        
        # Step 4: Validate with LiDAR data
        if lidar_data is not None and self.use_lidar_for_validation:
            camera_objects = self.validate_with_lidar(camera_objects, lidar_data)
        
        # Step 5: Update state estimate using Kalman filter
        if imu_data is not None:
            self.update_kalman_filter(imu_data, camera_objects)
        
        # Step 6: Calculate overall confidence in fused result
        confidence = self.calculate_fusion_confidence(camera_objects)
        
        fused_data['objects'] = camera_objects
        fused_data['confidence'] = confidence
        fused_data['robot_pose'] = self.estimate_robot_pose(imu_data)
        
        return fused_data
    
    def process_camera_image(self, image):
        """
        Process camera image to detect objects
        """
        # In a real implementation, this would use Isaac ROS detection nodes
        # For this example, we'll simulate object detection
        
        # Convert image to grayscale for simple processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simulate object detection (in reality, use Isaac ROS detection)
        objects = []
        
        # Example: detect bright spots as potential objects
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                
                # Create object detection result
                obj = {
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': {'x': x + w/2, 'y': y + h/2},
                    'confidence': 0.8,  # Simulated confidence
                    'class': 'bright_spot',  # Simulated class
                    'pixel_area': w * h
                }
                objects.append(obj)
        
        return objects
    
    def add_depth_information(self, objects, depth_image, camera_matrix):
        """
        Add depth information to detected objects
        """
        if not objects or depth_image is None:
            return objects
        
        # Calculate depth for each detected object
        for obj in objects:
            center_x = int(obj['center']['x'])
            center_y = int(obj['center']['y'])
            
            # Get depth value at object center (with bounds checking)
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]
                
                # Only add depth if it's valid (not NaN or infinity)
                if np.isfinite(depth) and depth > 0:
                    # Convert pixel coordinates to 3D world coordinates
                    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
                    
                    # Calculate 3D position
                    x_world = (center_x - cx) * depth / fx
                    y_world = (center_y - cy) * depth / fy
                    z_world = depth
                    
                    obj['position_3d'] = {'x': x_world, 'y': y_world, 'z': z_world}
                    obj['depth'] = depth
        
        return objects
    
    def correct_for_orientation(self, objects, imu_data):
        """
        Correct object positions based on IMU orientation data
        """
        if not objects or not imu_data:
            return objects
        
        # Extract orientation from IMU (quaternion)
        quat = imu_data['orientation']
        
        # Convert quaternion to rotation matrix
        rotation = R.from_quat(quat).as_matrix()
        
        # Apply rotation correction to 3D positions
        for obj in objects:
            if 'position_3d' in obj:
                pos_3d = np.array([obj['position_3d']['x'], 
                                 obj['position_3d']['y'], 
                                 obj['position_3d']['z']])
                
                # Apply rotation
                corrected_pos = rotation @ pos_3d
                
                obj['position_3d'] = {
                    'x': corrected_pos[0],
                    'y': corrected_pos[1], 
                    'z': corrected_pos[2]
                }
        
        return objects
    
    def validate_with_lidar(self, objects, lidar_data):
        """
        Validate camera detections with LiDAR data
        """
        if not objects or not lidar_data:
            return objects
        
        validated_objects = []
        
        for obj in objects:
            if 'position_3d' not in obj:
                validated_objects.append(obj)
                continue
            
            # Check if the object's 3D position corresponds to a LiDAR return
            object_pos = np.array([obj['position_3d']['x'], 
                                 obj['position_3d']['y'], 
                                 obj['position_3d']['z']])
            
            # Calculate range and bearing from robot to object
            range_to_object = np.linalg.norm(object_pos)
            bearing = np.arctan2(object_pos[1], object_pos[0])  # Azimuth angle
            
            # Find corresponding LiDAR measurement
            angle_index = int((bearing - lidar_data['angle_min']) / lidar_data['angle_increment'])
            
            if 0 <= angle_index < len(lidar_data['ranges']):
                lidar_range = lidar_data['ranges'][angle_index]
                
                # Validate if LiDAR range is consistent with camera depth
                if lidar_range > 0 and np.isfinite(lidar_range):
                    range_diff = abs(range_to_object - lidar_range)
                    
                    # If the difference is within acceptable threshold, validate the object
                    if range_diff < 0.2:  # 20cm tolerance
                        obj['lidar_validated'] = True
                        obj['lidar_range'] = lidar_range
                        obj['validation_confidence'] = max(0.5, obj['confidence'] * 0.9)  # Boost confidence if validated
                        validated_objects.append(obj)
                    else:
                        # If not validated, reduce confidence
                        obj['lidar_validated'] = False
                        obj['validation_confidence'] = obj['confidence'] * 0.7  # Reduce confidence
                        validated_objects.append(obj)
                else:
                    # No LiDAR return in that direction
                    obj['lidar_validated'] = False
                    obj['validation_confidence'] = obj['confidence'] * 0.8  # Slightly reduce confidence
                    validated_objects.append(obj)
            else:
                # Bearing out of LiDAR range
                obj['lidar_validated'] = False
                obj['validation_confidence'] = obj['confidence'] * 0.8
                validated_objects.append(obj)
        
        return validated_objects
    
    def update_kalman_filter(self, imu_data, camera_objects):
        """
        Update Kalman filter with sensor measurements
        """
        # This would implement the Kalman filter prediction and update steps
        # For this example, we'll just update the state based on IMU data
        
        if imu_data:
            # Extract linear acceleration (in robot frame)
            accel = np.array(imu_data['linear_acceleration'])
            
            # Integrate to get velocity and position
            dt = 0.01  # Assuming 100Hz IMU
            self.kalman_filter['state'][3:6] += accel * dt  # Update velocity
            self.kalman_filter['state'][0:3] += self.kalman_filter['state'][3:6] * dt  # Update position
    
    def calculate_fusion_confidence(self, objects):
        """
        Calculate overall confidence in the fused result
        """
        if not objects:
            return 0.0
        
        # Calculate confidence as average of individual object confidences
        total_confidence = sum(obj.get('validation_confidence', obj.get('confidence', 0.5)) for obj in objects)
        return total_confidence / len(objects) if objects else 0.0
    
    def estimate_robot_pose(self, imu_data):
        """
        Estimate robot pose from IMU data
        """
        if not imu_data:
            return None
        
        # Extract orientation from IMU
        orientation_quat = imu_data['orientation']
        
        # For position, we'd typically integrate velocity over time
        # For this example, we'll return the orientation
        return {
            'orientation': orientation_quat,
            'timestamp': imu_data['timestamp']
        }
    
    def publish_fused_data(self, fused_result, header):
        """
        Publish the fused sensor data
        """
        # Publish objects as Float32MultiArray (simplified)
        if fused_result['objects']:
            fused_msg = Float32MultiArray()
            fused_msg.layout.dim = [
                MultiArrayDimension(label="objects", size=len(fused_result['objects']), stride=6)
            ]
            
            # Pack object data (simplified - just position and confidence)
            for obj in fused_result['objects']:
                if 'position_3d' in obj:
                    pos = obj['position_3d']
                    fused_msg.data.extend([pos['x'], pos['y'], pos['z'], 
                                         obj.get('confidence', 0.5),
                                         obj.get('validation_confidence', 0.5),
                                         1.0 if obj.get('lidar_validated', False) else 0.0])
            
            fused_msg.header = header
            self.fused_perception_publisher.publish(fused_msg)
        
        # Publish robot pose
        if fused_result['robot_pose']:
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.pose.orientation.x = fused_result['robot_pose']['orientation'][0]
            pose_msg.pose.orientation.y = fused_result['robot_pose']['orientation'][1]
            pose_msg.pose.orientation.z = fused_result['robot_pose']['orientation'][2]
            pose_msg.pose.orientation.w = fused_result['robot_pose']['orientation'][3]
            self.fused_pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacROSSensorFusion()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac ROS Sensor Fusion Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with humanoid control

Integrating Isaac ROS with humanoid control systems requires careful coordination between the perception and control systems to ensure that the robot can react appropriately to visual information.

### Perception-Control Interface

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray
from humanoid_msgs.msg import GaitCommand, BalanceState
import numpy as np
from collections import deque
import threading
import time

class IsaacROSControlInterface(Node):
    def __init__(self):
        super().__init__('isaac_ros_control_interface')
        
        # Initialize perception and control components
        self.perception_pipeline = IsaacROSGPUAcceleratedPerception()
        self.balance_controller = None  # Will be initialized separately
        self.walk_controller = None     # Will be initialized separately
        
        # Create subscribers and publishers
        self.object_sub = self.create_subscription(
            Detection2DArray, '/isaac_ros/gpu_detections', self.object_detection_callback, 10
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        
        self.balance_state_sub = self.create_subscription(
            BalanceState, '/balance_state', self.balance_state_callback, 10
        )
        
        self.gait_command_pub = self.create_publisher(
            GaitCommand, '/gait_command', 10
        )
        
        self.walk_command_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )
        
        # Perception-control state
        self.current_objects = []
        self.joint_positions = {}
        self.balance_state = None
        self.is_active = True
        
        # Threading for perception-control coordination
        self.perception_lock = threading.Lock()
        self.control_lock = threading.Lock()
        
        # Buffer for object tracking
        self.object_track_buffer = deque(maxlen=10)
        
        # Parameters for perception-action coordination
        self.declare_parameter('detection_timeout', 0.5)
        self.declare_parameter('approach_threshold', 0.5)  # meters
        self.declare_parameter('avoidance_threshold', 0.3)  # meters
        self.declare_parameter('tracking_iou_threshold', 0.3)  # Intersection over Union threshold
        
        self.detection_timeout = self.get_parameter('detection_timeout').value
        self.approach_threshold = self.get_parameter('approach_threshold').value
        self.avoidance_threshold = self.get_parameter('avoidance_threshold').value
        self.tracking_iou_threshold = self.get_parameter('tracking_iou_threshold').value
        
        # Last detection time
        self.last_detection_time = self.get_clock().now()
        
        # Initialize controllers
        self.initialize_controllers()
        
        self.get_logger().info('Isaac ROS Control Interface initialized')
    
    def initialize_controllers(self):
        """
        Initialize balance and walking controllers
        """
        # This would typically connect to actual controller nodes
        # For this example, we'll create placeholder controllers
        
        # In a real system, you would initialize actual controllers:
        # self.balance_controller = BalanceController()
        # self.walk_controller = WalkingController()
        
        self.get_logger().info('Controllers initialized (placeholder)')
    
    def object_detection_callback(self, msg):
        """
        Process object detections and coordinate with control system
        """
        current_time = self.get_clock().now()
        
        with self.perception_lock:
            # Update object list
            self.current_objects = msg.detections
            self.last_detection_time = current_time
            
            # Track objects over time
            self.update_object_tracking(msg.detections)
        
        # Coordinate perception with control
        self.coordinate_perception_control()
    
    def update_object_tracking(self, detections):
        """
        Update object tracking with new detections
        """
        # Implement object tracking using IoU matching
        if not self.object_track_buffer:
            # First set of detections
            for det in detections:
                self.object_track_buffer.append({
                    'bbox': det.bbox,
                    'class': det.results[0].id if det.results else 'unknown',
                    'confidence': det.results[0].score if det.results else 0.0,
                    'timestamp': self.get_clock().now()
                })
            return
        
        # Match new detections with existing tracks
        updated_tracks = []
        for track in self.object_track_buffer:
            # Find best matching detection
            best_match_idx = -1
            best_iou = 0.0
            
            for i, det in enumerate(detections):
                iou = self.calculate_bbox_iou(track['bbox'], det.bbox)
                if iou > best_iou and iou > self.tracking_iou_threshold:
                    best_iou = iou
                    best_match_idx = i
        
            if best_match_idx >= 0:
                # Update track with new detection
                updated_track = {
                    'bbox': detections[best_match_idx].bbox,
                    'class': detections[best_match_idx].results[0].id if detections[best_match_idx].results else track['class'],
                    'confidence': detections[best_match_idx].results[0].score if detections[best_match_idx].results else track['confidence'],
                    'timestamp': self.get_clock().now()
                }
                updated_tracks.append(updated_track)
                
                # Remove matched detection from list
                detections.pop(best_match_idx)
            else:
                # Track not found - may be lost or out of view
                if (self.get_clock().now() - track['timestamp']).nanoseconds / 1e9 < 1.0:  # 1 second timeout
                    updated_tracks.append(track)  # Keep for a short time
        
        # Add new detections as new tracks
        for det in detections:
            new_track = {
                'bbox': det.bbox,
                'class': det.results[0].id if det.results else 'unknown',
                'confidence': det.results[0].score if det.results else 0.0,
                'timestamp': self.get_clock().now()
            }
            updated_tracks.append(new_track)
        
        self.object_track_buffer = deque(updated_tracks, maxlen=10)
    
    def calculate_bbox_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        """
        # Extract center and size
        x1_c, y1_c = bbox1.center.x, bbox1.center.y
        w1, h1 = bbox1.size_x, bbox1.size_y
        
        x2_c, y2_c = bbox2.center.x, bbox2.center.y
        w2, h2 = bbox2.size_x, bbox2.size_y
        
        # Calculate corners
        x1_min, y1_min = x1_c - w1/2, y1_c - h1/2
        x1_max, y1_max = x1_c + w1/2, y1_c + h1/2
        
        x2_min, y2_min = x2_c - w2/2, y2_c - h2/2
        x2_max, y2_max = x2_c + w2/2, y2_c + h2/2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calculate union area
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        else:
            return 0.0
    
    def joint_state_callback(self, msg):
        """
        Update joint state information
        """
        with self.control_lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
    
    def balance_state_callback(self, msg):
        """
        Update balance state information
        """
        self.balance_state = msg
    
    def coordinate_perception_control(self):
        """
        Coordinate perception and control based on detected objects
        """
        if not self.current_objects:
            # No objects detected, continue normal walking
            return
        
        # Process each detected object to determine appropriate action
        for detection in self.current_objects:
            if detection.results and len(detection.results) > 0:
                obj_class = detection.results[0].id
                confidence = detection.results[0].score
                bbox = detection.bbox
                
                if confidence > 0.7:  # High confidence detection
                    action_required = self.evaluate_object_action(obj_class, bbox)
                    
                    if action_required:
                        self.execute_perception_driven_action(action_required)
    
    def evaluate_object_action(self, obj_class, bbox):
        """
        Evaluate what action should be taken based on detected object
        """
        # Convert bounding box center to approximate distance (simplified)
        # In a real system, this would use depth information
        center_x = bbox.center.x
        center_y = bbox.center.y
        size_x = bbox.size_x
        size_y = bbox.size_y
        
        # Estimate distance based on object size (assuming known object size)
        # This is a simplified approach - in reality, use depth information
        estimated_distance = self.estimate_distance_from_size(obj_class, size_x, size_y)
        
        # Determine action based on object class and distance
        if obj_class == 'person':
            if estimated_distance < self.approach_threshold:
                # Person is close, stop or move aside
                return {
                    'action': 'avoid_person',
                    'direction': 'left' if center_x < 320 else 'right',  # Assuming 640x480 image
                    'distance': estimated_distance
                }
            elif estimated_distance < self.approach_threshold * 2:
                # Person is in approach range, slow down
                return {
                    'action': 'slow_down',
                    'factor': 0.5,
                    'distance': estimated_distance
                }
        elif obj_class == 'obstacle' or obj_class == 'chair':
            if estimated_distance < self.avoidance_threshold:
                # Obstacle is too close, avoid
                return {
                    'action': 'avoid_obstacle',
                    'direction': 'left' if center_x < 320 else 'right',
                    'distance': estimated_distance
                }
        elif obj_class == 'target_object':
            # Object of interest - approach if close enough
            if estimated_distance < self.approach_threshold:
                return {
                    'action': 'approach_object',
                    'bbox': bbox,
                    'distance': estimated_distance
                }
        
        return None
    
    def estimate_distance_from_size(self, obj_class, size_x, size_y):
        """
        Estimate distance based on object size in image (simplified approach)
        """
        # This is a simplified estimation - in reality, use depth camera data
        # For this example, assume known object size and use size-distance relationship
        known_sizes = {
            'person': 1.7,  # meters
            'chair': 0.5,
            'table': 0.8,
            'cup': 0.1,
            'bottle': 0.25
        }
        
        # Calculate average size
        avg_size_pixels = (size_x + size_y) / 2
        
        # Convert pixel size to angle (simplified)
        # Assuming 60 degree FOV for 640 pixels: 60/640 = 0.09375 degrees per pixel
        pixel_to_angle = 0.09375  # degrees per pixel
        object_angle = avg_size_pixels * pixel_to_angle * np.pi / 180  # radians
        
        # Estimate distance using known size and angle
        if obj_class in known_sizes:
            known_size = known_sizes[obj_class]
            distance = known_size / (2 * np.tan(object_angle / 2)) if object_angle > 0 else float('inf')
            return min(distance, 5.0)  # Cap at 5 meters
        else:
            # Unknown object - use a default approach
            return 1.0  # Default to 1 meter
    
    def execute_perception_driven_action(self, action):
        """
        Execute action based on perception input
        """
        action_type = action['action']
        
        if action_type == 'avoid_person':
            self.execute_person_avoidance(action)
        elif action_type == 'slow_down':
            self.execute_speed_reduction(action)
        elif action_type == 'avoid_obstacle':
            self.execute_obstacle_avoidance(action)
        elif action_type == 'approach_object':
            self.execute_object_approach(action)
    
    def execute_person_avoidance(self, action):
        """
        Execute action to avoid a person
        """
        direction = action['direction']
        distance = action['distance']
        
        # Stop walking temporarily
        stop_cmd = Twist()
        self.walk_command_publisher.publish(stop_cmd)
        
        # Issue gait command to step aside
        gait_cmd = GaitCommand()
        gait_cmd.command = 'step'
        gait_cmd.direction = 'left' if direction == 'left' else 'right'
        gait_cmd.step_size = 0.2  # meters
        gait_cmd.speed = 0.3  # m/s
        
        self.gait_command_publisher.publish(gait_cmd)
        
        self.get_logger().info(f'Avoding person {distance:.2f}m away by stepping {direction}')
    
    def execute_speed_reduction(self, action):
        """
        Execute action to reduce walking speed
        """
        factor = action['factor']
        
        # Reduce current walking speed by factor
        current_cmd = Twist()  # In a real system, this would be the current command
        current_cmd.linear.x *= factor
        current_cmd.angular.z *= factor
        
        self.walk_command_publisher.publish(current_cmd)
        
        self.get_logger().info(f'Reduced walking speed by factor of {factor}')
    
    def execute_obstacle_avoidance(self, action):
        """
        Execute action to avoid an obstacle
        """
        direction = action['direction']
        distance = action['distance']
        
        # Stop walking
        stop_cmd = Twist()
        self.walk_command_publisher.publish(stop_cmd)
        
        # Issue avoidance command
        gait_cmd = GaitCommand()
        gait_cmd.command = 'step'
        gait_cmd.direction = 'left' if direction == 'left' else 'right'
        gait_cmd.step_size = 0.3  # meters
        gait_cmd.speed = 0.4  # m/s
        
        self.gait_command_publisher.publish(gait_cmd)
        
        self.get_logger().info(f'Avoiding obstacle {distance:.2f}m away by stepping {direction}')
    
    def execute_object_approach(self, action):
        """
        Execute action to approach a detected object
        """
        bbox = action['bbox']
        distance = action['distance']
        
        # Calculate approach direction based on object position
        image_center_x = 320  # Assuming 640x480 image
        object_center_x = bbox.center.x
        
        # Determine if we need to turn
        if abs(object_center_x - image_center_x) > 50:  # 50 pixel threshold
            # Turn toward object
            turn_cmd = Twist()
            turn_cmd.angular.z = 0.3 if object_center_x < image_center_x else -0.3  # Turn left or right
            self.walk_command_publisher.publish(turn_cmd)
        else:
            # Move forward to approach object
            approach_cmd = Twist()
            approach_cmd.linear.x = 0.2  # Move forward slowly
            self.walk_command_publisher.publish(approach_cmd)
        
        self.get_logger().info(f'Approaching object {distance:.2f}m away')
    
    def check_detection_timeout(self):
        """
        Check if we've had no detections for too long
        """
        current_time = self.get_clock().now()
        time_since_detection = (current_time - self.last_detection_time).nanoseconds / 1e9
        
        if time_since_detection > self.detection_timeout:
            self.get_logger().debug('No recent detections, continuing default behavior')
    
    def run_perception_control_loop(self):
        """
        Main loop for perception-control coordination
        """
        while rclpy.ok() and self.is_active:
            # Check for detection timeouts
            self.check_detection_timeout()
            
            # Process any pending perception-control coordination
            # This would be handled by callbacks in a real implementation
            
            time.sleep(0.01)  # 100Hz loop

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacROSControlInterface()
    
    try:
        # Run the perception-control loop in a separate thread
        control_thread = threading.Thread(target=node.run_perception_control_loop)
        control_thread.start()
        
        # Spin the node to handle callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac ROS Control Interface')
        node.is_active = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Conclusion

Isaac ROS provides a powerful platform for integrating hardware-accelerated perception with humanoid robot control systems. By leveraging GPU computing, TensorRT optimization, and advanced sensor fusion techniques, robots can process sensor data in real-time and make intelligent decisions about navigation, manipulation, and interaction.

The integration of Isaac ROS with humanoid control systems enables robots to perceive and respond to their environment dynamically, making them more capable of operating in complex, unstructured environments. As Isaac ROS continues to evolve, we can expect even more sophisticated perception capabilities that will further enhance the abilities of humanoid robots.