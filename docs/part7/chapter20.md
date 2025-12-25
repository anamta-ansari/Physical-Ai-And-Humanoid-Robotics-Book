---
title: Edge Computing (Jetson Orin Nano/NX)
sidebar_position: 2
description: Deploying ROS nodes on Jetson platforms, sensor integration, and edge AI for humanoid robotics
---

# Edge Computing (Jetson Orin Nano/NX)

## Why edge AI matters

Edge AI has become increasingly important in humanoid robotics, providing the computational power needed for real-time perception, decision-making, and control while maintaining low latency and reduced dependence on cloud connectivity. NVIDIA's Jetson platform offers an ideal solution for deploying AI models directly on robots.

### The Need for Edge Computing in Robotics

Traditional robotics systems often rely on cloud-based processing, which introduces several challenges:

1. **Latency**: Communication delays can be critical for real-time control
2. **Bandwidth**: High-resolution sensor data requires significant bandwidth
3. **Connectivity**: Robots must function in areas with poor or no network connection
4. **Privacy**: Sensitive data may need to remain on the robot
5. **Reliability**: Cloud services may be unavailable when needed
6. **Cost**: Continuous cloud usage can be expensive

### Advantages of Edge AI for Humanoid Robots

#### Low Latency Response
- **Real-time Control**: Immediate response to sensor inputs
- **Safety**: Faster reaction to potential hazards
- **Smooth Operation**: Continuous, uninterrupted robot behavior

#### Reduced Bandwidth Requirements
- **On-device Processing**: Compute happens locally
- **Selective Transmission**: Only send processed results, not raw data
- **Offline Capability**: Function without network connection

#### Privacy and Security
- **Local Data Processing**: Sensitive information stays on robot
- **Reduced Attack Surface**: Less network exposure
- **Compliance**: Meets privacy regulations for data processing

#### Cost Efficiency
- **Reduced Cloud Costs**: No per-use fees for AI processing
- **Scalability**: Deploy many robots without increasing cloud costs
- **Operational Savings**: Lower data transmission costs

### Edge AI vs Cloud AI Comparison

| Aspect | Edge AI | Cloud AI |
|--------|---------|----------|
| **Latency** | {'<'}10ms | 50-200ms+ |
| **Bandwidth** | Minimal | High |
| **Connectivity** | Not required | Required |
| **Cost** | Upfront + maintenance | Ongoing usage fees |
| **Processing Power** | Limited by device | Virtually unlimited |
| **Privacy** | High (local) | Lower (transmitted) |
| **Reliability** | High (local) | Dependent on cloud |

### Edge AI in Humanoid Robotics Applications

#### Perception Tasks
- **Object Detection**: Identify people, obstacles, and targets
- **SLAM**: Simultaneous Localization and Mapping
- **Gesture Recognition**: Understanding human gestures
- **Facial Recognition**: Identifying specific individuals
- **Scene Understanding**: Context awareness

#### Control Tasks
- **Gait Control**: Real-time adjustment of walking patterns
- **Balance Control**: Maintaining stability in dynamic environments
- **Path Planning**: On-the-fly navigation adjustments
- **Manipulation**: Real-time grasp planning and control

#### Decision Making
- **Behavior Selection**: Choosing appropriate responses
- **Task Planning**: Sequencing complex actions
- **Social Interaction**: Appropriate responses to humans

## Deploying ROS nodes on Jetson

Deploying ROS nodes on Jetson platforms requires understanding the constraints and capabilities of the edge platform while maintaining compatibility with the broader ROS ecosystem.

### Jetson Platform Overview

#### Jetson Orin Nano
- **GPU**: 1024-core NVIDIA Ampere architecture GPU
- **CPU**: 4-core ARM Cortex-A78AE v8.2 64-bit CPU
- **Memory**: 4GB or 8GB LPDDR5
- **DL TOPS**: 27 TOPS (int8), 54 TOPS (int4)
- **Power**: 15W-25W
- **Form Factor**: Compact module

#### Jetson Orin NX
- **GPU**: 1024-core NVIDIA Ampere architecture GPU
- **CPU**: 6-core ARM Cortex-A78AE v8.2 64-bit CPU
- **Memory**: 8GB or 16GB LPDDR5
- **DL TOPS**: 77 TOPS (int8), 154 TOPS (int4)
- **Power**: 15W-25W
- **Performance**: Higher than Orin Nano

### Setting Up ROS 2 on Jetson

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ROS 2 Humble Hawksbill
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release

# Add ROS 2 repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
sudo apt install -y python3-rosdep python3-vcstool

# Initialize rosdep
sudo rosdep init
rosdep update

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Optimized ROS Nodes for Edge Deployment

```python
#!/usr/bin/env python3
# Optimized ROS node for Jetson edge computing
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import time
import threading
from functools import partial

class JetsonOptimizedPerceptionNode(Node):
    def __init__(self):
        super().__init__('jetson_edge_perception')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1  # Low queue size to reduce memory usage
        )
        
        # Create publishers
        self.detection_pub = self.create_publisher(String, '/edge_detections', 1)
        self.performance_pub = self.create_publisher(String, '/performance_metrics', 1)
        
        # Optimization parameters
        self.declare_parameter('processing_rate', 10)  # Hz
        self.declare_parameter('image_resize_factor', 0.5)  # Reduce image size
        self.declare_parameter('enable_gpu_acceleration', True)
        self.declare_parameter('max_memory_usage_mb', 1000)  # Limit memory usage
        
        self.processing_rate = self.get_parameter('processing_rate').value
        self.resize_factor = self.get_parameter('image_resize_factor').value
        self.enable_gpu_acceleration = self.get_parameter('enable_gpu_acceleration').value
        self.max_memory_mb = self.get_parameter('max_memory_usage_mb').value
        
        # Initialize optimized models
        self.initialize_optimized_models()
        
        # Threading for efficient processing
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_process_time = time.time()
        
        # Memory monitoring
        self.memory_monitor = self.create_timer(1.0, self.monitor_memory)
        
        self.get_logger().info('Jetson Edge Perception Node initialized')
    
    def initialize_optimized_models(self):
        """
        Initialize optimized models suitable for Jetson edge computing
        """
        try:
            # For Jetson, we'll use TensorRT optimized models or INT8 quantized models
            if self.enable_gpu_acceleration:
                import jetson.inference
                import jetson.utils
                
                # Initialize optimized detection model
                # This would typically load a TensorRT optimized model
                self.detection_model = self.load_jetson_optimized_model()
                
                self.get_logger().info('GPU-accelerated model loaded for Jetson')
            else:
                # Fallback to CPU model
                import cv2
                self.detection_model = cv2.dnn.readNetFromONNX('optimized_model.onnx')
                
                self.get_logger().info('CPU model loaded')
        
        except ImportError as e:
            self.get_logger().warn(f'Could not load optimized models: {e}')
            self.get_logger().info('Using basic OpenCV implementation')
            import cv2
            self.detection_model = cv2.dnn.readNetFromONNX('basic_model.onnx') if False else None  # Placeholder
    
    def load_jetson_optimized_model(self):
        """
        Load Jetson-optimized model using TensorRT
        """
        # This is a conceptual implementation
        # In practice, you'd use jetson.inference or similar
        return {
            'model_loaded': True,
            'tensorrt_optimized': True,
            'precision': 'int8',  # Quantized for efficiency
            'input_size': (416, 416)  # Optimized input size for edge
        }
    
    def image_callback(self, msg):
        """
        Process incoming image with optimization for edge computing
        """
        current_time = time.time()
        
        # Throttle processing rate to prevent overwhelming the system
        if current_time - self.last_process_time < 1.0 / self.processing_rate:
            return  # Skip frame if processing too fast
        
        # Add to processing queue (non-blocking)
        with self.queue_lock:
            # Limit queue size to prevent memory buildup
            if len(self.processing_queue) < 3:
                self.processing_queue.append(msg)
            else:
                # Drop oldest if queue too long
                self.processing_queue.pop(0)
                self.processing_queue.append(msg)
    
    def process_queue(self):
        """
        Process images from the queue in a separate thread
        """
        while rclpy.ok():
            with self.queue_lock:
                if len(self.processing_queue) > 0:
                    msg = self.processing_queue.pop(0)
                else:
                    time.sleep(0.01)  # Sleep briefly if no work
                    continue
            
            try:
                # Process the image
                self.process_image_optimized(msg)
                
                # Update performance metrics
                self.frame_count += 1
                self.last_process_time = time.time()
                
            except Exception as e:
                self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_image_optimized(self, msg):
        """
        Optimized image processing for Jetson edge platform
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Resize image to reduce computational load
            if self.resize_factor != 1.0:
                new_width = int(cv_image.shape[1] * self.resize_factor)
                new_height = int(cv_image.shape[0] * self.resize_factor)
                cv_image = cv2.resize(cv_image, (new_width, new_height))
            
            # Preprocess for model (optimized for edge)
            input_tensor = self.preprocess_for_edge_model(cv_image)
            
            # Run inference (optimized for Jetson)
            if self.detection_model:
                start_time = time.time()
                
                if self.enable_gpu_acceleration and self.detection_model.get('model_loaded'):
                    # Use optimized GPU inference
                    results = self.run_jetson_optimized_inference(input_tensor)
                else:
                    # Use CPU inference
                    results = self.run_cpu_inference(input_tensor)
                
                inference_time = time.time() - start_time
                
                # Process results
                processed_results = self.process_detection_results(results, cv_image.shape)
                
                # Publish results
                self.publish_detection_results(processed_results, msg.header)
                
                # Publish performance metrics
                self.publish_performance_metrics(inference_time, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error in optimized processing: {str(e)}')
    
    def preprocess_for_edge_model(self, image):
        """
        Preprocess image for edge-optimized model
        """
        # Resize to model input size (optimized for edge)
        input_size = (416, 416)  # Common size for optimized models
        resized_image = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Normalize using ImageNet statistics (but optimized for INT8)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized_image = (rgb_image / 255.0 - mean) / std
        
        # Change to CHW format
        input_tensor = np.transpose(normalized_image, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor.astype(np.float32)
    
    def run_jetson_optimized_inference(self, input_tensor):
        """
        Run inference using Jetson-optimized model
        """
        # This would interface with jetson.inference or TensorRT
        # For this example, we'll simulate optimized inference
        try:
            # Simulate optimized inference results
            # In practice, this would call the actual optimized model
            return {
                'detections': [
                    {'class': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 300]},
                    {'class': 'chair', 'confidence': 0.72, 'bbox': [250, 200, 350, 350]}
                ],
                'processing_time': 0.02  # 20ms processing time
            }
        except Exception as e:
            self.get_logger().error(f'Error in Jetson optimized inference: {str(e)}')
            return {'detections': [], 'processing_time': 0.1}  # Fallback with longer time
    
    def run_cpu_inference(self, input_tensor):
        """
        Fallback CPU inference
        """
        # Use OpenCV DNN for CPU inference
        blob = cv2.dnn.blobFromImage(input_tensor[0], 1.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        
        # This is a simplified example - in practice, you'd use the loaded model
        return {
            'detections': [
                {'class': 'person', 'confidence': 0.75, 'bbox': [110, 110, 190, 290]},
                {'class': 'table', 'confidence': 0.68, 'bbox': [260, 210, 340, 340]}
            ],
            'processing_time': 0.15  # 150ms processing time
        }
    
    def process_detection_results(self, results, image_shape):
        """
        Process detection results and format for ROS publication
        """
        height, width = image_shape[:2]
        
        processed_detections = []
        
        for detection in results['detections']:
            # Adjust bounding box coordinates to original image size if resized
            if self.resize_factor != 1.0:
                x1, y1, x2, y2 = detection['bbox']
                detection['bbox'] = [
                    int(x1 / self.resize_factor),
                    int(y1 / self.resize_factor),
                    int(x2 / self.resize_factor),
                    int(y2 / self.resize_factor)
                ]
            
            processed_detections.append(detection)
        
        return {
            'detections': processed_detections,
            'image_resolution': [width, height],
            'processing_time': results['processing_time']
        }
    
    def publish_detection_results(self, results, header):
        """
        Publish detection results
        """
        result_msg = String()
        result_msg.data = str(results)
        self.detection_publisher.publish(result_msg)
    
    def publish_performance_metrics(self, inference_time, header):
        """
        Publish performance metrics
        """
        # Calculate metrics
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            'current_fps': 1.0 / inference_time if inference_time > 0 else 0,
            'average_fps': avg_fps,
            'inference_time_ms': inference_time * 1000,
            'memory_usage_mb': self.get_current_memory_usage(),
            'processing_rate': self.processing_rate
        }
        
        metrics_msg = String()
        metrics_msg.data = str(metrics)
        self.performance_publisher.publish(metrics_msg)
    
    def monitor_memory(self):
        """
        Monitor memory usage and adjust processing if needed
        """
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 85:  # High memory usage
                self.get_logger().warn(f'High memory usage: {memory_percent:.1f}%')
                
                # Reduce processing rate to conserve memory
                if self.processing_rate > 5:  # Don't go below 5Hz
                    self.processing_rate = max(5, self.processing_rate * 0.8)
                    self.get_logger().info(f'Reduced processing rate to {self.processing_rate}Hz due to memory pressure')
            
            elif memory_percent < 60 and self.processing_rate < 20:  # Low memory usage, can increase rate
                self.processing_rate = min(20, self.processing_rate * 1.1)
                self.get_logger().info(f'Increased processing rate to {self.processing_rate}Hz')
                
        except ImportError:
            # psutil not available, skip memory monitoring
            pass
    
    def get_current_memory_usage(self):
        """
        Get current memory usage
        """
        try:
            import psutil
            return psutil.virtual_memory().used / (1024**2)  # Convert to MB
        except ImportError:
            return 0  # Return 0 if psutil not available

def main(args=None):
    rclpy.init(args=args)
    
    node = JetsonOptimizedPerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Jetson Edge Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Jetson-Specific Optimizations

```python
class JetsonHardwareOptimizations:
    """
    Class containing Jetson-specific hardware optimizations
    """
    def __init__(self):
        self.jetson_model = self.detect_jetson_model()
        self.power_mode = 'MAXN'  # MAXN or 5W for power saving
        self.fan_control_enabled = True
        
        # Initialize Jetson-specific optimizations
        self.setup_jetson_optimizations()
    
    def detect_jetson_model(self):
        """
        Detect the specific Jetson model
        """
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                return model
        except:
            return "Unknown Jetson Model"
    
    def setup_jetson_optimizations(self):
        """
        Set up Jetson-specific optimizations
        """
        self.get_logger().info(f'Jetson Model: {self.jetson_model}')
        
        # Configure power mode
        self.configure_power_mode()
        
        # Set up fan control
        self.setup_fan_control()
        
        # Optimize memory management
        self.optimize_memory_management()
        
        # Configure GPU settings
        self.configure_gpu_settings()
    
    def configure_power_mode(self):
        """
        Configure Jetson power mode for optimal performance
        """
        try:
            import subprocess
            if 'Orin' in self.jetson_model:
                # For Jetson Orin devices
                subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)  # MAXN mode
                subprocess.run(['sudo', 'jetson_clocks'], check=True)  # Enable max clocks
                self.get_logger().info('Configured Jetson Orin for MAXN power mode')
            elif 'Nano' in self.jetson_model:
                # For Jetson Nano
                subprocess.run(['sudo', 'nvpmodel', '-m', '0'], check=True)  # MAXN mode
                self.get_logger().info('Configured Jetson Nano for MAXN power mode')
        except Exception as e:
            self.get_logger().warn(f'Could not configure power mode: {str(e)}')
    
    def setup_fan_control(self):
        """
        Set up fan control for thermal management
        """
        if not self.fan_control_enabled:
            return
        
        try:
            # Example fan control script (this would be more complex in reality)
            import subprocess
            subprocess.run(['sudo', 'pwmconfig'], check=True)  # Configure PWM for fan
        except Exception as e:
            self.get_logger().warn(f'Could not configure fan control: {str(e)}')
    
    def optimize_memory_management(self):
        """
        Optimize memory management for edge AI
        """
        # Configure memory for AI workloads
        try:
            # Increase shared memory size for large tensors
            import os
            shm_size = 2 * 1024 * 1024 * 1024  # 2GB
            os.system(f'sudo mount -o remount,size={shm_size} /dev/shm')
            self.get_logger().info('Increased shared memory for AI workloads')
        except Exception as e:
            self.get_logger().warn(f'Could not optimize memory: {str(e)}')
    
    def configure_gpu_settings(self):
        """
        Configure GPU settings for optimal AI performance
        """
        try:
            # Set GPU to maximum performance mode
            import subprocess
            # For Jetson devices, this might involve setting GPU frequency
            subprocess.run(['sudo', 'nvpmodel', '-q'], check=True, capture_output=True)
            self.get_logger().info('GPU configured for optimal performance')
        except Exception as e:
            self.get_logger().warn(f'Could not configure GPU settings: {str(e)}')
    
    def optimize_for_latency_critical_tasks(self):
        """
        Optimize system for low-latency robotics tasks
        """
        # Configure real-time scheduling
        try:
            import os
            # Set process to real-time priority (requires proper system configuration)
            os.system('sudo chrt -f 99 $$')  # Set to FIFO real-time scheduling
        except Exception as e:
            self.get_logger().warn(f'Could not set real-time priority: {str(e)}')
        
        # Optimize I/O scheduler
        try:
            with open('/sys/block/mmcblk0/queue/scheduler', 'w') as f:
                f.write('deadline\n')  # Use deadline scheduler for robotics
        except Exception as e:
            self.get_logger().warn(f'Could not optimize I/O scheduler: {str(e)}')
    
    def get_jetson_performance_stats(self):
        """
        Get performance statistics specific to Jetson platform
        """
        stats = {}
        
        try:
            # Get Jetson stats using jetson-stats package if available
            import subprocess
            result = subprocess.run(['jtop', '-c', '1'], capture_output=True, text=True)
            if result.returncode == 0:
                stats['jetson_stats'] = result.stdout
            else:
                # Alternative: get basic stats manually
                stats['temperature'] = self.get_temperature()
                stats['gpu_utilization'] = self.get_gpu_utilization()
                stats['cpu_utilization'] = self.get_cpu_utilization()
        except Exception as e:
            self.get_logger().warn(f'Could not get Jetson stats: {str(e)}')
            stats['error'] = str(e)
        
        return stats
    
    def get_temperature(self):
        """
        Get system temperature
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read().strip()) / 1000.0  # Convert to Celsius
                return temp
        except:
            return 0.0
    
    def get_gpu_utilization(self):
        """
        Get GPU utilization
        """
        try:
            # This is a simplified example - in practice, you'd use jetson-stats
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except:
            return 0
    
    def get_cpu_utilization(self):
        """
        Get CPU utilization
        """
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0

# Example launch file for Jetson deployment
"""
# jetson_edge_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    processing_rate = LaunchConfiguration('processing_rate')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'processing_rate',
            default_value='10',
            description='Processing rate for perception nodes (Hz)'
        ),
        
        # Set environment variables for Jetson optimization
        SetEnvironmentVariable(
            name='CUDA_VISIBLE_DEVICES',
            value='0'
        ),
        SetEnvironmentVariable(
            name='TF_CPP_MIN_LOG_LEVEL',
            value='2'  # Reduce TensorFlow logging
        ),
        
        # Jetson-optimized perception node
        Node(
            package='jetson_perception',
            executable='jetson_optimized_perception',
            name='jetson_edge_perception',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'processing_rate': processing_rate},
                {'image_resize_factor': 0.5},  # Optimize for Jetson
                {'enable_gpu_acceleration': True}
            ],
            remappings=[
                ('/camera/image_raw', '/usb_camera/image_raw'),
                ('/edge_detections', '/robot/detections')
            ]
        ),
        
        # Jetson-optimized control node
        Node(
            package='jetson_control',
            executable='jetson_optimized_controller',
            name='jetson_edge_controller',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'control_frequency': 100},  # Higher for real-time control
                {'enable_gpu_planning': True}
            ]
        ),
        
        # Jetson-specific system monitoring
        Node(
            package='jetson_system_monitor',
            executable='jetson_health_monitor',
            name='jetson_health_monitor',
            parameters=[
                {'thermal_warning_threshold': 75.0},
                {'memory_warning_threshold': 85.0}
            ]
        )
    ])
"""
```

## Sensor integration: RealSense D435i, IMU, Microphones

Integrating sensors effectively on edge platforms requires optimizing for both computational efficiency and sensor fusion accuracy.

### RealSense D435i Integration

The Intel RealSense D435i is an excellent choice for humanoid robots as it provides both RGB and depth information along with integrated IMU data.

```python
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import Image, Imu, PointCloud2
from cv_bridge import CvBridge
import threading
import queue

class RealSenseD435iNode(Node):
    def __init__(self):
        super().__init__('realsense_d435i_node')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Get device intrinsics
        profile = self.pipeline.get_active_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.color_intrinsics = color_profile.get_intrinsics()
        
        # Create publishers
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/camera/imu', 200)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/camera/depth/points', 10)
        
        # Processing parameters
        self.enable_pointcloud = True
        self.pointcloud_decimation = 2  # Reduce point cloud density for performance
        self.publish_rate = 30  # Hz
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.get_logger().info('RealSense D435i node initialized')
    
    def capture_loop(self):
        """
        Main capture loop running in separate thread
        """
        while rclpy.ok():
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                
                # Process depth frame
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
                    depth_msg.header.stamp = self.get_clock().now().to_msg()
                    depth_msg.header.frame_id = 'camera_depth_optical_frame'
                    self.depth_publisher.publish(depth_msg)
                
                # Process color frame
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    color_msg = self.cv_bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                    color_msg.header.stamp = self.get_clock().now().to_msg()
                    color_msg.header.frame_id = 'camera_color_optical_frame'
                    self.color_publisher.publish(color_msg)
                
                # Process IMU frames
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                
                if accel_frame and gyro_frame:
                    self.publish_imu_data(accel_frame, gyro_frame)
                
                # Process point cloud if enabled
                if self.enable_pointcloud and depth_frame and color_frame:
                    self.publish_pointcloud(depth_frame, color_frame)
                
            except Exception as e:
                self.get_logger().error(f'Error in capture loop: {str(e)}')
    
    def publish_imu_data(self, accel_frame, gyro_frame):
        """
        Publish IMU data from RealSense
        """
        try:
            # Get IMU data
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            
            # Create IMU message
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'camera_imu_frame'
            
            # Set angular velocity (gyro data)
            imu_msg.angular_velocity.x = gyro_data.x
            imu_msg.angular_velocity.y = gyro_data.y
            imu_msg.angular_velocity.z = gyro_data.z
            
            # Set linear acceleration (accelerometer data)
            imu_msg.linear_acceleration.x = accel_data.x
            imu_msg.linear_acceleration.y = accel_data.y
            imu_msg.linear_acceleration.z = accel_data.z
            
            # Note: RealSense D435i doesn't provide orientation data directly
            # We would need to integrate the gyro data or use a separate IMU
            # For now, set orientation to unknown
            imu_msg.orientation_covariance[0] = -1  # Indicates no orientation data
            
            self.imu_publisher.publish(imu_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing IMU data: {str(e)}')
    
    def publish_pointcloud(self, depth_frame, color_frame):
        """
        Publish point cloud from depth and color data
        """
        try:
            # Convert depth frame to point cloud
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get point cloud
            points = rs.pointcloud()
            pc_map = points.calculate(depth_frame)
            
            # Extract vertices
            vertices = np.asanyarray(pc_map.get_vertices(2))
            
            # Create PointCloud2 message
            from sensor_msgs_py import point_cloud2
            from std_msgs.msg import Header
            
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'camera_depth_optical_frame'
            
            # Decimate points for performance
            if self.pointcloud_decimation > 1:
                vertices = vertices[::self.pointcloud_decimation]
            
            # Create point cloud
            # Define point fields (x, y, z)
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            
            # Filter out invalid points (where depth is 0)
            valid_points = vertices[vertices[:, 2] > 0]  # Only points with valid depth
            
            if len(valid_points) > 0:
                pointcloud_msg = point_cloud2.create_cloud(header, fields, valid_points)
                self.pointcloud_publisher.publish(pointcloud_msg)
        
        except Exception as e:
            self.get_logger().error(f'Error publishing point cloud: {str(e)}')
    
    def destroy_node(self):
        """
        Clean up RealSense resources
        """
        self.pipeline.stop()
        super().destroy_node()

# Example usage of the RealSense node
def main(args=None):
    rclpy.init(args=args)
    
    node = RealSenseD435iNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down RealSense node')
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### IMU Integration

Integrating IMU data is crucial for balance and orientation in humanoid robots:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3
import numpy as np
import threading
import time

class IMUIntegrationNode(Node):
    def __init__(self):
        super().__init__('imu_integration_node')
        
        # Create publishers
        self.imu_pub = self.create_publisher(Imu, '/robot/imu/data', 200)
        self.mag_pub = self.create_publisher(MagneticField, '/robot/imu/mag', 200)
        
        # Create subscriber for raw IMU data
        self.raw_imu_sub = self.create_subscription(
            Imu, '/camera/imu', self.raw_imu_callback, 200
        )
        
        # IMU fusion parameters
        self.complementary_filter_alpha = 0.98
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Quaternion [x, y, z, w]
        self.previous_time = None
        self.bias = np.zeros(3)  # Gyroscope bias
        
        # Initialize bias estimation
        self.bias_samples = []
        self.bias_sample_count = 100  # Number of samples to estimate bias
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_samples = 0
        
        # Create timer for publishing fused IMU data
        self.imu_timer = self.create_timer(0.01, self.publish_fused_imu)  # 100Hz
        
        self.get_logger().info('IMU Integration Node initialized')
    
    def raw_imu_callback(self, msg):
        """
        Process raw IMU data and perform sensor fusion
        """
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        
        if self.previous_time is None:
            self.previous_time = current_time
            return
        
        dt = current_time - self.previous_time
        self.previous_time = current_time
        
        # Extract IMU data
        accel = np.array([msg.linear_acceleration.x, 
                         msg.linear_acceleration.y, 
                         msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, 
                        msg.angular_velocity.y, 
                        msg.angular_velocity.z])
        
        # Calibrate gyroscope bias if not calibrated
        if not self.is_calibrated:
            self.calibrate_gyro_bias(gyro)
            return
        
        # Remove bias from gyroscope readings
        gyro_corrected = gyro - self.bias
        
        # Perform complementary filter fusion
        self.update_orientation_complementary(accel, gyro_corrected, dt)
    
    def calibrate_gyro_bias(self, gyro_reading):
        """
        Calibrate gyroscope bias by taking average of stationary readings
        """
        if self.calibration_samples < self.bias_sample_count:
            self.bias_samples.append(gyro_reading.copy())
            self.calibration_samples += 1
        else:
            # Calculate bias as average of samples
            self.bias = np.mean(self.bias_samples, axis=0)
            self.is_calibrated = True
            self.get_logger().info(f'Gyroscope bias calibrated: {self.bias}')
    
    def update_orientation_complementary(self, accel, gyro, dt):
        """
        Update orientation using complementary filter
        """
        # Integrate gyroscope data for short-term orientation
        gyro_rotation = self.integrate_gyro(gyro, dt)
        
        # Calculate orientation from accelerometer (gravity vector)
        accel_orientation = self.calculate_orientation_from_accel(accel)
        
        # Fuse accelerometer and gyroscope data using complementary filter
        # Apply low-pass filter to accelerometer data
        alpha = self.complementary_filter_alpha
        
        # Update orientation using complementary filter
        self.orientation = self.quaternion_slerp(
            self.orientation, 
            accel_orientation, 
            1 - alpha
        )
        
        # Apply gyroscope integration
        gyro_quat = self.axis_angle_to_quaternion(gyro * dt)
        self.orientation = self.quaternion_multiply(self.orientation, gyro_quat)
        
        # Normalize quaternion to prevent drift
        self.orientation = self.normalize_quaternion(self.orientation)
    
    def integrate_gyro(self, gyro, dt):
        """
        Integrate gyroscope data to get rotation
        """
        # Convert angular velocity to axis-angle representation
        angle = np.linalg.norm(gyro) * dt
        
        if angle < 1e-6:  # Avoid division by zero
            return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        
        axis = gyro / np.linalg.norm(gyro)
        
        # Convert to quaternion
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle)
        ])
    
    def calculate_orientation_from_accel(self, accel):
        """
        Calculate orientation from accelerometer data (gravity vector)
        """
        # Normalize accelerometer data
        accel_norm = accel / np.linalg.norm(accel) if np.linalg.norm(accel) > 0 else np.array([0, 0, 1])
        
        # Calculate roll and pitch from gravity vector
        pitch = np.arctan2(-accel_norm[0], np.sqrt(accel_norm[1]**2 + accel_norm[2]**2))
        roll = np.arctan2(accel_norm[1], accel_norm[2])
        
        # Convert to quaternion
        cy = np.cos(yaw * 0.5)  # Assuming yaw from magnetometer
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([x, y, z, w])
    
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])
    
    def normalize_quaternion(self, q):
        """
        Normalize a quaternion
        """
        norm = np.linalg.norm(q)
        if norm > 0:
            return q / norm
        else:
            return np.array([0, 0, 0, 1])  # Identity quaternion
    
    def quaternion_slerp(self, q1, q2, t):
        """
        Spherical linear interpolation between two quaternions
        """
        # Calculate dot product
        dot = np.dot(q1, q2)
        
        # If dot product is negative, negate one quaternion
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return self.normalize_quaternion(result)
        
        # Calculate angle between quaternions
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def axis_angle_to_quaternion(self, axis_angle):
        """
        Convert axis-angle representation to quaternion
        """
        angle = np.linalg.norm(axis_angle)
        
        if angle < 1e-6:  # Very small angle
            return np.array([0, 0, 0, 1])
        
        axis = axis_angle / angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_angle)
        ])
    
    def publish_fused_imu(self):
        """
        Publish fused IMU data
        """
        if not self.is_calibrated:
            return
        
        # Create and publish fused IMU message
        fused_imu_msg = Imu()
        fused_imu_msg.header.stamp = self.get_clock().now().to_msg()
        fused_imu_msg.header.frame_id = 'imu_link'
        
        # Set orientation
        fused_imu_msg.orientation.x = float(self.orientation[0])
        fused_imu_msg.orientation.y = float(self.orientation[1])
        fused_imu_msg.orientation.z = float(self.orientation[2])
        fused_imu_msg.orientation.w = float(self.orientation[3])
        
        # Set orientation covariance (indicates accuracy)
        # Set to 0.01^2 = 0.0001 for each diagonal element
        fused_imu_msg.orientation_covariance = [0.0001] * 9
        
        # Set angular velocity (this would come from raw IMU with bias correction)
        # For now, we'll leave it as 0 since we're computing orientation
        # In practice, you'd have access to corrected gyro data
        
        # Set linear acceleration (would come from raw IMU with gravity removed)
        # For now, we'll leave it as 0
        
        self.imu_publisher.publish(fused_imu_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = IMUIntegrationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down IMU Integration Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### Microphone Integration

Integrating microphones enables voice interaction capabilities for humanoid robots:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
import pyaudio
import numpy as np
import threading
import webrtcvad
import collections

class MicrophoneNode(Node):
    def __init__(self):
        super().__init__('microphone_node')
        
        # Audio parameters
        self.sample_rate = 16000  # Hz
        self.chunk_size = 1024  # Frames per chunk
        self.channels = 1  # Mono
        self.format = pyaudio.paInt16  # 16-bit samples
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Audio buffer for processing
        self.audio_buffer = collections.deque(maxlen=int(self.sample_rate * 2))  # 2 seconds buffer
        self.voice_activity_buffer = collections.deque(maxlen=100)  # For voice activity history
        
        # Create publisher for audio data
        self.audio_pub = self.create_publisher(AudioData, '/robot/audio/raw', 10)
        self.speech_pub = self.create_publisher(String, '/robot/speech', 10)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self.audio_capture_loop, daemon=True)
        self.audio_thread.start()
        
        # Start voice activity detection thread
        self.vad_thread = threading.Thread(target=self.voice_activity_detection_loop, daemon=True)
        self.vad_thread.start()
        
        self.get_logger().info(f'Microphone node initialized with sample rate: {self.sample_rate}Hz')
    
    def audio_capture_loop(self):
        """
        Audio capture loop running in separate thread
        """
        # Open audio stream
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        try:
            while rclpy.ok():
                # Read audio chunk
                audio_chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk)
                
                # Publish audio data
                audio_msg = AudioData()
                audio_msg.data = audio_chunk
                self.audio_publisher.publish(audio_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error in audio capture: {str(e)}')
        finally:
            stream.stop_stream()
            stream.close()
    
    def voice_activity_detection_loop(self):
        """
        Voice activity detection running in separate thread
        """
        while rclpy.ok():
            # Check if we have enough audio data for VAD
            if len(self.audio_buffer) >= self.chunk_size * 2:  # Need at least 2 chunks
                # Extract audio data for VAD
                vad_data = bytes(list(self.audio_buffer)[:self.chunk_size*2])
                
                # Run VAD on 10ms frames
                frame_size = int(self.sample_rate * 0.01)  # 10ms frame
                if len(vad_data) >= frame_size * 2:
                    # Check first frame for voice activity
                    frame = vad_data[:frame_size*2]
                    
                    try:
                        # VAD expects 10ms, 20ms, or 30ms frames
                        if len(frame) == frame_size * 2:  # 20ms
                            is_speech = self.vad.is_speech(frame, self.sample_rate)
                            
                            # Add to voice activity history
                            self.voice_activity_buffer.append(is_speech)
                            
                            # Check if speech is detected
                            if is_speech and not self.is_recently_speaking():
                                self.get_logger().info('Voice activity detected!')
                                
                                # Publish speech notification
                                speech_msg = String()
                                speech_msg.data = 'voice_detected'
                                self.speech_publisher.publish(speech_msg)
                        
                        time.sleep(0.01)  # Process every 10ms
                    except Exception as e:
                        self.get_logger().warn(f'VAD error: {str(e)}')
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
    
    def is_recently_speaking(self):
        """
        Check if speech was recently detected
        """
        # Count recent voice activity
        recent_activity = sum(list(self.voice_activity_buffer)[-10:])  # Last 10 detections
        return recent_activity > 3  # If 3+ of last 10 indicate speech
    
    def get_audio_features(self, audio_data):
        """
        Extract audio features for speech recognition
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0  # Normalize to [-1, 1]
        
        # Extract features
        features = {
            'rms_energy': np.sqrt(np.mean(audio_array**2)),
            'zero_crossing_rate': np.sum(np.diff(np.sign(audio_array)) != 0) / len(audio_array),
            'spectral_centroid': self.calculate_spectral_centroid(audio_array),
            'mfccs': self.calculate_mfccs(audio_array)
        }
        
        return features
    
    def calculate_spectral_centroid(self, audio_array):
        """
        Calculate spectral centroid (brightness of sound)
        """
        # Compute FFT
        fft = np.fft.fft(audio_array)
        magnitude = np.abs(fft[:len(fft)//2])  # Take positive frequencies
        
        # Calculate spectral centroid
        freqs = np.arange(len(magnitude))
        centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        return centroid
    
    def calculate_mfccs(self, audio_array, num_mfccs=13):
        """
        Calculate Mel-Frequency Cepstral Coefficients
        """
        try:
            import librosa
            mfccs = librosa.feature.mfcc(
                y=audio_array, 
                sr=self.sample_rate, 
                n_mfcc=num_mfccs
            )
            return mfccs.mean(axis=1)  # Return mean across time
        except ImportError:
            # Fallback if librosa not available
            return np.zeros(num_mfccs)
    
    def destroy_node(self):
        """
        Clean up audio resources
        """
        self.audio.terminate()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = MicrophoneNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Microphone Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

## Edge computing optimization

Optimizing for edge computing involves reducing computational demands while maintaining performance.

### Model Optimization Techniques

```python
import tensorflow as tf
import numpy as np

class EdgeModelOptimizer:
    def __init__(self):
        self.optimizer = tf.lite.Optimize.DEFAULT
        self.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops if needed
        ]
    
    def optimize_model_for_jetson(self, model_path, output_path):
        """
        Optimize a TensorFlow model for Jetson deployment
        """
        # Load the model
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set supported operations
        converter.target_spec.supported_ops = self.supported_ops
        
        # Enable experimental optimizations for NVIDIA hardware
        converter.experimental_new_converter = True
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the optimized model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Optimized model saved to {output_path}")
        return output_path
    
    def quantize_model(self, model_path, output_path, quantization_type='int8'):
        """
        Quantize model to reduce size and improve inference speed
        """
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        
        if quantization_type == 'int8':
            # INT8 quantization with calibration
            def representative_dataset():
                for _ in range(100):
                    # Generate representative data for calibration
                    data = np.random.random((1, 224, 224, 3)).astype(np.float32)
                    yield [data]
            
            converter.representative_dataset = representative_dataset
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        elif quantization_type == 'float16':
            # FLOAT16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Quantized model saved to {output_path}")
        return output_path
    
    def prune_model(self, model, sparsity=0.5):
        """
        Prune model to remove redundant connections
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            # Define pruning parameters
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            # Apply pruning to the model
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
            
            # Compile the model
            model_for_pruning.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model_for_pruning
        except ImportError:
            print("TensorFlow Model Optimization Toolkit not available")
            return model

# Example usage
optimizer = EdgeModelOptimizer()

# Optimize a model for Jetson deployment
# optimized_model_path = optimizer.optimize_model_for_jetson(
#     'path/to/original/model', 
#     'path/to/optimized/model.tflite'
# )

# Quantize the model
# quantized_model_path = optimizer.quantize_model(
#     'path/to/original/model', 
#     'path/to/quantized/model.tflite', 
#     'int8'
# )
```

### Resource Management

```python
import psutil
import GPUtil
import subprocess
import time

class ResourceManager:
    def __init__(self):
        self.max_cpu_usage = 80.0  # Percentage
        self.max_memory_usage = 80.0  # Percentage
        self.max_gpu_usage = 85.0  # Percentage
        self.max_temperature = 75.0  # Celsius
        
        # Performance scaling parameters
        self.cpu_scaling_enabled = True
        self.gpu_scaling_enabled = True
        self.dynamic_batching_enabled = True
        
        # Resource monitoring
        self.monitoring_interval = 1.0  # seconds
        self.resource_history = {
            'cpu': [],
            'memory': [],
            'gpu': [],
            'temperature': []
        }
    
    def monitor_resources(self):
        """
        Monitor system resources and adjust performance accordingly
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory_percent = psutil.virtual_memory().percent
        
        # Get GPU usage (if available)
        gpu_percent = 0
        gpu_memory_percent = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_memory_percent = gpus[0].memoryUtil * 100
        except:
            pass
        
        # Get system temperature
        temp_celsius = self.get_system_temperature()
        
        # Store in history
        self.resource_history['cpu'].append(cpu_percent)
        self.resource_history['memory'].append(memory_percent)
        self.resource_history['gpu'].append(gpu_percent)
        self.resource_history['temperature'].append(temp_celsius)
        
        # Keep history to last 100 samples
        for key in self.resource_history:
            if len(self.resource_history[key]) > 100:
                self.resource_history[key].pop(0)
        
        # Check for resource overuse and take corrective action
        if cpu_percent > self.max_cpu_usage:
            self.handle_cpu_overuse()
        
        if memory_percent > self.max_memory_usage:
            self.handle_memory_overuse()
        
        if gpu_percent > self.max_gpu_usage:
            self.handle_gpu_overuse()
        
        if temp_celsius > self.max_temperature:
            self.handle_overheating()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_percent': gpu_percent,
            'gpu_memory_percent': gpu_memory_percent,
            'temperature': temp_celsius
        }
    
    def get_system_temperature(self):
        """
        Get system temperature
        """
        try:
            # Try different methods to get temperature
            # Method 1: Using sensors (Linux)
            result = subprocess.run(['sensors'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse temperature from sensors output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Core' in line and '+' in line and '°C' in line:
                        temp_str = line.split('+')[1].split('°C')[0].strip()
                        try:
                            return float(temp_str)
                        except ValueError:
                            continue
            
            # Method 2: Try reading from thermal zone
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0  # Convert from millidegrees
                    return temp
            except:
                pass
            
            # If all methods fail, return 0
            return 0.0
        except:
            return 0.0
    
    def handle_cpu_overuse(self):
        """
        Handle excessive CPU usage
        """
        self.get_logger().warn(f'High CPU usage detected: {self.resource_history["cpu"][-1]:.1f}%')
        
        # Reduce processing rate
        if self.processing_rate > 5:  # Don't go below 5Hz
            self.processing_rate = max(5, self.processing_rate * 0.9)
            self.get_logger().info(f'Reduced processing rate to {self.processing_rate}Hz due to CPU pressure')
    
    def handle_memory_overuse(self):
        """
        Handle excessive memory usage
        """
        self.get_logger().warn(f'High memory usage detected: {self.resource_history["memory"][-1]:.1f}%')
        
        # Clear caches and reduce buffer sizes
        try:
            # Clear system caches (requires sudo)
            subprocess.run(['sudo', 'sh', '-c', 'echo 1 > /proc/sys/vm/drop_caches'], check=True)
        except:
            pass  # Continue if cache clearing fails
        
        # Reduce buffer sizes in perception nodes
        self.reduce_buffer_sizes()
    
    def handle_gpu_overuse(self):
        """
        Handle excessive GPU usage
        """
        self.get_logger().warn(f'High GPU usage detected: {self.resource_history["gpu"][-1]:.1f}%')
        
        # Reduce model complexity or batch size
        self.reduce_gpu_workload()
    
    def handle_overheating(self):
        """
        Handle overheating conditions
        """
        temp = self.resource_history['temperature'][-1]
        self.get_logger().warn(f'Overheating detected: {temp:.1f}°C')
        
        # Reduce performance to cool down
        if self.processing_rate > 2:  # Minimum processing rate
            self.processing_rate = max(2, self.processing_rate * 0.8)
            self.get_logger().info(f'Reduced processing rate to {self.processing_rate}Hz to reduce heat')
        
        # Increase fan speed if possible
        self.increase_fan_speed()
    
    def reduce_buffer_sizes(self):
        """
        Reduce buffer sizes to save memory
        """
        # This would adjust buffer sizes in perception nodes
        # For example, reducing image queue sizes, point cloud buffer sizes, etc.
        pass
    
    def reduce_gpu_workload(self):
        """
        Reduce GPU workload
        """
        # This would involve reducing model resolution, batch size, or complexity
        pass
    
    def increase_fan_speed(self):
        """
        Increase fan speed for cooling
        """
        try:
            # Example: Increase fan speed via PWM control
            # This would be platform-specific
            pass
        except:
            # Fallback: reduce processing rate
            self.processing_rate = max(1, self.processing_rate * 0.7)
            self.get_logger().info(f'Reduced processing rate to {self.processing_rate}Hz due to overheating')
    
    def get_resource_recommendations(self):
        """
        Get recommendations for optimizing resource usage
        """
        avg_cpu = np.mean(self.resource_history['cpu'][-10:]) if self.resource_history['cpu'] else 0
        avg_memory = np.mean(self.resource_history['memory'][-10:]) if self.resource_history['memory'] else 0
        avg_gpu = np.mean(self.resource_history['gpu'][-10:]) if self.resource_history['gpu'] else 0
        max_temp = max(self.resource_history['temperature'][-10:]) if self.resource_history['temperature'] else 0
        
        recommendations = []
        
        if avg_cpu > 70:
            recommendations.append("Consider optimizing CPU-intensive algorithms or reducing processing rate")
        
        if avg_memory > 70:
            recommendations.append("Consider using more memory-efficient data structures or reducing buffer sizes")
        
        if avg_gpu > 75:
            recommendations.append("Consider using quantized models or reducing resolution")
        
        if max_temp > 70:
            recommendations.append("Consider improving cooling or reducing computational load")
        
        return recommendations

# Example usage in a ROS node
class EdgeOptimizedNode(Node):
    def __init__(self):
        super().__init__('edge_optimized_node')
        
        # Initialize resource manager
        self.resource_manager = ResourceManager()
        
        # Start resource monitoring
        self.resource_timer = self.create_timer(1.0, self.monitor_resources)
        
        # Performance adjustment parameters
        self.processing_rate = 10  # Hz
        self.current_batch_size = 1
        self.model_complexity = 'medium'  # 'low', 'medium', 'high'
    
    def monitor_resources(self):
        """
        Monitor resources and adjust performance
        """
        resources = self.resource_manager.monitor_resources()
        
        # Log resource usage
        self.get_logger().debug(
            f'Resources - CPU: {resources["cpu_percent"]:.1f}%, '
            f'Mem: {resources["memory_percent"]:.1f}%, '
            f'GPU: {resources["gpu_percent"]:.1f}%, '
            f'Temp: {resources["temperature"]:.1f}°C'
        )
        
        # Get recommendations
        recommendations = self.resource_manager.get_resource_recommendations()
        for rec in recommendations:
            self.get_logger().info(f'Resource recommendation: {rec}')
```

## Conclusion

Edge computing with platforms like NVIDIA Jetson enables humanoid robots to perform complex AI tasks directly on the robot, reducing latency and dependence on cloud connectivity. Proper integration of sensors like the RealSense D435i, IMU, and microphones with optimized processing creates a capable perception system. Resource management and model optimization ensure that these systems can run efficiently on the constrained hardware of edge platforms.

The combination of efficient ROS nodes, optimized AI models, and proper sensor fusion enables humanoid robots to operate autonomously in real-world environments while maintaining responsive and reliable performance.