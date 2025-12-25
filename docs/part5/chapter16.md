---
title: ROS 2 Integration
sidebar_position: 5
description: ROS 2 architecture, pub/sub, services, actions, launch files, parameters, and best practices for humanoid robotics
---

# ROS 2 Integration

## ROS 2 architecture

ROS 2 (Robot Operating System 2) is the next-generation robotics middleware that provides a framework for developing robot applications. It addresses many of the limitations of ROS 1, particularly around real-time performance, security, and multi-robot systems.

### Core Architecture Components

ROS 2 uses a DDS (Data Distribution Service) based architecture that provides:

1. **Node**: A process that performs computation
2. **Topic**: Named bus over which nodes exchange messages
3. **Service**: Synchronous request/reply communication
4. **Action**: Asynchronous goal-based communication with feedback
5. **Parameter**: Configuration values that can be changed at runtime

### DDS Implementation Options

ROS 2 supports multiple DDS implementations:
- **Fast DDS** (formerly Fast RTPS): eProsima's implementation (default in newer ROS 2 versions)
- **Cyclone DDS**: Eclipse Foundation's implementation
- **RTI Connext DDS**: RTI's commercial implementation
- **OpenSplice DDS**: ADLINK's implementation

### Communication Patterns

ROS 2 supports several communication patterns:

#### Publisher/Subscriber (Pub/Sub)
- Asynchronous, many-to-many communication
- Messages are published to topics
- Subscribers receive messages from topics they're interested in

#### Services
- Synchronous, one-to-one communication
- Client sends request, server responds with reply

#### Actions
- Asynchronous, goal-oriented communication
- Includes feedback during execution and result upon completion

### Quality of Service (QoS)

QoS settings allow fine-tuning of communication behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# Example QoS profiles for different use cases

# Real-time control (e.g., joint commands)
control_qos = QoSProfile(
    depth=1,  # Only keep most recent message
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure delivery
    durability=DurabilityPolicy.VOLATILE,  # Don't keep old messages
    history=HistoryPolicy.KEEP_LAST  # Keep only last N messages
)

# Sensor data (e.g., camera images)
sensor_qos = QoSProfile(
    depth=5,  # Keep several messages
    reliability=ReliabilityPolicy.BEST_EFFORT,  # OK to lose some messages
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

# Configuration data (e.g., robot parameters)
config_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,  # Keep for late-joining nodes
    history=HistoryPolicy.KEEP_LAST
)
```

### ROS 2 Node Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class HumanoidRobotNode(Node):
    def __init__(self):
        super().__init__('humanoid_robot_node')
        
        # Declare parameters
        self.declare_parameter('robot_name', 'unitree_g1')
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_margin', 0.1)
        
        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_margin = self.get_parameter('safety_margin').value
        
        # Create publishers
        self.joint_command_publisher = self.create_publisher(
            JointState, 
            f'/{self.robot_name}/joint_commands', 
            10
        )
        
        self.base_velocity_publisher = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            10
        )
        
        self.status_publisher = self.create_publisher(
            String,
            f'/{self.robot_name}/status',
            10
        )
        
        # Create subscribers
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.image_subscriber = self.create_subscription(
            Image,
            f'/{self.robot_name}/camera/image_raw',
            self.image_callback,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT
            )
        )
        
        # Create service server
        self.reset_service = self.create_service(
            SetBool,
            f'/{self.robot_name}/reset',
            self.reset_callback
        )
        
        # Create action server
        self.fibonacci_action_server = ActionServer(
            self,
            Fibonacci,
            f'/{self.robot_name}/fibonacci',
            self.execute_fibonacci_callback
        )
        
        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0/self.control_frequency,
            self.control_loop_callback
        )
        
        # Initialize robot state
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.is_running = True
        
        self.get_logger().info(f'Humanoid Robot Node initialized for {self.robot_name}')
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]
    
    def image_callback(self, msg):
        """
        Callback for image messages
        """
        # Process image data (in a real implementation, this would be more complex)
        self.get_logger().debug(f'Received image with dimensions: {msg.width}x{msg.height}')
    
    def reset_callback(self, request, response):
        """
        Callback for reset service
        """
        if request.data:
            self.get_logger().info('Resetting robot...')
            # Perform reset operations
            self.reset_robot_state()
            response.success = True
            response.message = 'Robot reset successfully'
        else:
            response.success = False
            response.message = 'Reset request rejected'
        
        return response
    
    def execute_fibonacci_callback(self, goal_handle):
        """
        Execute callback for Fibonacci action
        """
        self.get_logger().info('Executing Fibonacci action...')
        
        # Feedback and result
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()
        
        # Initialize sequence
        fibonacci_sequence = [0, 1]
        
        for i in range(1, goal_handle.request.order):
            # Check if preempted
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.sequence = fibonacci_sequence
                return result_msg
            
            # Update sequence
            next_fib = fibonacci_sequence[i] + fibonacci_sequence[i-1]
            fibonacci_sequence.append(next_fib)
            
            # Publish feedback
            feedback_msg.sequence = fibonacci_sequence
            goal_handle.publish_feedback(feedback_msg)
            
            # Sleep to simulate processing time
            time.sleep(0.1)
        
        # Complete goal
        goal_handle.succeed()
        result_msg.sequence = fibonacci_sequence
        return result_msg
    
    def control_loop_callback(self):
        """
        Main control loop
        """
        if not self.is_running:
            return
        
        # Perform control calculations
        self.update_robot_control()
        
        # Publish status
        status_msg = String()
        status_msg.data = f'Running - {len(self.joint_positions)} joints updated'
        self.status_publisher.publish(status_msg)
    
    def update_robot_control(self):
        """
        Update robot control based on current state
        """
        # This is where the actual control algorithm would run
        # For example, implementing walking patterns, balance control, etc.
        pass
    
    def reset_robot_state(self):
        """
        Reset robot to safe state
        """
        # Reset joint positions to default
        default_positions = {
            'left_hip_yaw': 0.0, 'left_hip_roll': 0.0, 'left_hip_pitch': 0.1,
            'left_knee_pitch': -0.5, 'left_ankle_pitch': 0.1, 'left_ankle_roll': 0.0,
            'right_hip_yaw': 0.0, 'right_hip_roll': 0.0, 'right_hip_pitch': 0.1,
            'right_knee_pitch': -0.5, 'right_ankle_pitch': 0.1, 'right_ankle_roll': 0.0,
            # Add other joints as needed
        }
        
        # Send reset commands
        reset_msg = JointState()
        reset_msg.name = list(default_positions.keys())
        reset_msg.position = list(default_positions.values())
        
        self.joint_command_publisher.publish(reset_msg)
    
    def destroy_node(self):
        """
        Cleanup when node is destroyed
        """
        self.get_logger().info('Shutting down Humanoid Robot Node...')
        self.is_running = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    robot_node = HumanoidRobotNode()
    
    try:
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        robot_node.get_logger().info('Interrupted by user')
    finally:
        robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers and subscribers

Publishers and subscribers form the backbone of ROS 2's communication system, enabling asynchronous message passing between nodes.

### Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time
import numpy as np
import time

class RobotPublisherNode(Node):
    def __init__(self):
        super().__init__('robot_publisher_node')
        
        # Create publishers for different robot data streams
        self.joint_state_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.imu_publisher = self.create_publisher(Imu, '/imu/data', 10)
        self.camera_publisher = self.create_publisher(Image, '/camera/image_raw', 1)
        self.foot_pressure_publisher = self.create_publisher(Float64MultiArray, '/foot_pressure', 10)
        self.com_publisher = self.create_publisher(PointStamped, '/center_of_mass', 10)
        
        # Timer for publishing data at specific frequencies
        self.joint_publish_timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz
        self.imu_publish_timer = self.create_timer(0.005, self.publish_imu_data)      # 200 Hz
        self.camera_publish_timer = self.create_timer(0.033, self.publish_camera_data) # ~30 Hz
        self.pressure_publish_timer = self.create_timer(0.02, self.publish_pressure_data) # 50 Hz
        self.com_publish_timer = self.create_timer(0.01, self.publish_com_data)       # 100 Hz
        
        # Robot state simulation
        self.simulated_joint_positions = np.zeros(24)  # 24 DoF humanoid
        self.simulated_joint_velocities = np.zeros(24)
        self.simulated_joint_efforts = np.zeros(24)
        self.time_offset = time.time()
        
        # Joint names for the humanoid robot
        self.joint_names = [
            # Left leg
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            # Right leg
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll',
            # Left arm
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_wrist_pitch', 'left_wrist_yaw',
            # Right arm
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_wrist_pitch', 'right_wrist_yaw',
            # Neck
            'neck_yaw', 'neck_pitch'
        ]
        
        self.get_logger().info('Robot Publisher Node initialized')
    
    def publish_joint_states(self):
        """
        Publish joint state messages
        """
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Simulate joint position changes
        dt = 0.01  # 10ms time step
        self.simulated_joint_positions += self.simulated_joint_velocities * dt
        
        # Add some oscillation to simulate walking
        phase = (time.time() - self.time_offset) * 2 * np.pi * 0.5  # 0.5 Hz oscillation
        for i in range(len(self.simulated_joint_positions)):
            # Add walking-like oscillation to leg joints
            if 'hip' in self.joint_names[i] or 'knee' in self.joint_names[i] or 'ankle' in self.joint_names[i]:
                self.simulated_joint_positions[i] += 0.1 * np.sin(phase + i * 0.2)
        
        msg.name = self.joint_names
        msg.position = self.simulated_joint_positions.tolist()
        msg.velocity = self.simulated_joint_velocities.tolist()
        msg.effort = self.simulated_joint_efforts.tolist()
        
        self.joint_state_publisher.publish(msg)
    
    def publish_imu_data(self):
        """
        Publish IMU data
        """
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'imu_link'
        
        # Simulate IMU data with some realistic values
        current_time = time.time()
        phase = (current_time - self.time_offset) * 2 * np.pi * 1.0  # 1 Hz oscillation
        
        # Simulate body oscillation during walking
        msg.orientation.x = 0.01 * np.sin(phase)
        msg.orientation.y = 0.02 * np.cos(phase * 0.7)  # Different frequency for balance
        msg.orientation.z = 0.005 * np.sin(phase * 1.3)
        msg.orientation.w = np.sqrt(max(0, 1 - (msg.orientation.x**2 + 
                                               msg.orientation.y**2 + 
                                               msg.orientation.z**2)))
        
        # Angular velocity (derivative of orientation)
        msg.angular_velocity.x = 0.01 * np.cos(phase) * 2 * np.pi * 1.0
        msg.angular_velocity.y = 0.02 * np.sin(phase * 0.7) * 2 * np.pi * 0.7 * (-1)
        msg.angular_velocity.z = 0.005 * np.cos(phase * 1.3) * 2 * np.pi * 1.3
        
        # Linear acceleration (with gravity component)
        msg.linear_acceleration.x = 0.5 * np.sin(phase * 2)  # Forward-back sway
        msg.linear_acceleration.y = 0.3 * np.cos(phase * 1.5)  # Side-to-side sway
        msg.linear_acceleration.z = 9.81 + 0.2 * np.sin(phase * 3)  # Gravity + small oscillation
        
        # Add realistic noise
        noise_level = 0.001
        msg.orientation.x += np.random.normal(0, noise_level)
        msg.orientation.y += np.random.normal(0, noise_level)
        msg.orientation.z += np.random.normal(0, noise_level)
        msg.angular_velocity.x += np.random.normal(0, noise_level * 10)
        msg.angular_velocity.y += np.random.normal(0, noise_level * 10)
        msg.angular_velocity.z += np.random.normal(0, noise_level * 10)
        msg.linear_acceleration.x += np.random.normal(0, noise_level * 5)
        msg.linear_acceleration.y += np.random.normal(0, noise_level * 5)
        msg.linear_acceleration.z += np.random.normal(0, noise_level * 5)
        
        self.imu_publisher.publish(msg)
    
    def publish_camera_data(self):
        """
        Publish camera image data
        """
        # In a real implementation, this would come from an actual camera
        # For simulation, we'll create a dummy image
        width, height = 640, 480
        channels = 3  # RGB
        
        # Create a simulated image (in reality, this would be captured from camera)
        image_data = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
        
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height = height
        msg.width = width
        msg.encoding = 'rgb8'
        msg.is_bigendian = False
        msg.step = width * channels  # Bytes per row
        msg.data = image_data.tobytes()
        
        self.camera_publisher.publish(msg)
    
    def publish_pressure_data(self):
        """
        Publish foot pressure sensor data
        """
        msg = Float64MultiArray()
        msg.layout.dim = [
            # Define 4 pressure sensors per foot
            # Each foot: front-left, front-right, rear-left, rear-right
        ]
        
        # Simulate pressure distribution during walking
        current_time = time.time()
        gait_phase = (current_time - self.time_offset) % 2.0  # 2-second gait cycle
        
        # Pressure values (in Newtons)
        left_foot_pressures = [0.0, 0.0, 0.0, 0.0]
        right_foot_pressures = [0.0, 0.0, 0.0, 0.0]
        
        # Simulate gait pattern
        if gait_phase < 1.0:
            # Left foot stance phase
            weight_distribution = min(gait_phase * 2.0, 1.0)  # 0 to 1
            left_foot_pressures = [200 * weight_distribution] * 4  # Even distribution
            right_foot_pressures = [0.0] * 4  # Right foot lifted
        else:
            # Right foot stance phase
            weight_distribution = min((gait_phase - 1.0) * 2.0, 1.0)  # 0 to 1
            right_foot_pressures = [200 * weight_distribution] * 4  # Even distribution
            left_foot_pressures = [0.0] * 4  # Left foot lifted
        
        # Add some variation to make it more realistic
        for i in range(4):
            left_foot_pressures[i] += np.random.normal(0, 5)
            right_foot_pressures[i] += np.random.normal(0, 5)
        
        # Ensure pressures don't go negative
        left_foot_pressures = [max(0, p) for p in left_foot_pressures]
        right_foot_pressures = [max(0, p) for p in right_foot_pressures]
        
        msg.data = left_foot_pressures + right_foot_pressures
        
        self.foot_pressure_publisher.publish(msg)
    
    def publish_com_data(self):
        """
        Publish center of mass data
        """
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Simulate CoM position during walking
        current_time = time.time()
        phase = (current_time - self.time_offset) * 2 * np.pi * 0.5  # 0.5 Hz walking
        
        # CoM oscillates laterally and vertically during walking
        msg.point.x = 0.0  # Average forward position
        msg.point.y = 0.05 * np.sin(phase)  # Lateral sway
        msg.point.z = 0.85 + 0.02 * np.sin(phase * 2)  # Vertical oscillation
        
        self.com_publisher.publish(msg)

# Example usage
def main(args=None):
    rclpy.init(args=args)
    
    publisher_node = RobotPublisherNode()
    
    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        publisher_node.get_logger().info('Interrupted by user')
    finally:
        publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Image
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float64MultiArray, String
from std_msgs.msg import Bool
import numpy as np
import cv2
from cv_bridge import CvBridge

class RobotSubscriberNode(Node):
    def __init__(self):
        super().__init__('robot_subscriber_node')
        
        # Initialize CvBridge for image processing
        self.cv_bridge = CvBridge()
        
        # Create subscribers for robot data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        self.camera_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            1  # Low QoS for camera to avoid overwhelming
        )
        
        self.pressure_subscriber = self.create_subscription(
            Float64MultiArray,
            '/foot_pressure',
            self.pressure_callback,
            10
        )
        
        self.com_subscriber = self.create_subscription(
            PointStamped,
            '/center_of_mass',
            self.com_callback,
            10
        )
        
        # Create publisher for processed data
        self.balance_command_publisher = self.create_publisher(Twist, '/balance_cmd', 10)
        self.status_publisher = self.create_publisher(String, '/subscriber_status', 10)
        
        # Robot state tracking
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.imu_data = {}
        self.camera_image = None
        self.foot_pressures = {'left': [0]*4, 'right': [0]*4}
        self.com_position = np.array([0.0, 0.0, 0.0])
        
        # Balance control parameters
        self.com_error_integral = np.zeros(2)  # For PID control
        self.com_error_derivative = np.zeros(2)
        self.prev_com_error = np.zeros(2)
        self.balance_pid_gains = {'kp': 50.0, 'ki': 0.1, 'kd': 10.0}
        
        # Update timer for balance control
        self.balance_timer = self.create_timer(0.01, self.balance_control_callback)  # 100 Hz
        
        self.get_logger().info('Robot Subscriber Node initialized')
    
    def joint_state_callback(self, msg):
        """
        Callback for joint state messages
        """
        # Update internal joint state
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]
        
        # Check for unusual joint states (potential problems)
        for joint_name, position in self.joint_positions.items():
            if abs(position) > 3.0:  # Check for potentially dangerous positions
                self.get_logger().warn(f'Unusual joint position for {joint_name}: {position}')
    
    def imu_callback(self, msg):
        """
        Callback for IMU data
        """
        # Store IMU data
        self.imu_data = {
            'orientation': np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]),
            'angular_velocity': np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]),
            'linear_acceleration': np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        }
        
        # Check for balance issues based on IMU data
        orientation_norm = np.linalg.norm(self.imu_data['orientation'][:3])
        if abs(orientation_norm - 1.0) > 0.1:
            self.get_logger().warn('IMU orientation quaternion not normalized')
        
        # Check for excessive angular velocity (indicating potential fall)
        angular_vel_norm = np.linalg.norm(self.imu_data['angular_velocity'])
        if angular_vel_norm > 1.0:  # Threshold for excessive rotation
            self.get_logger().warn(f'Excessive angular velocity: {angular_vel_norm}')
    
    def camera_callback(self, msg):
        """
        Callback for camera images
        """
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Store image for processing
            self.camera_image = cv_image
            
            # Perform basic image processing if needed
            self.process_camera_image(cv_image)
            
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {str(e)}')
    
    def pressure_callback(self, msg):
        """
        Callback for foot pressure sensor data
        """
        if len(msg.data) >= 8:  # Expecting 4 sensors per foot
            self.foot_pressures['left'] = msg.data[:4]
            self.foot_pressures['right'] = msg.data[4:8]
        
        # Check for balance based on pressure distribution
        self.analyze_pressure_distribution()
    
    def com_callback(self, msg):
        """
        Callback for center of mass data
        """
        self.com_position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
        # Check if CoM is outside safe bounds
        if abs(self.com_position[1]) > 0.15:  # Lateral CoM too far
            self.get_logger().warn(f'Lateral CoM displacement too large: {self.com_position[1]}')
    
    def process_camera_image(self, image):
        """
        Process camera image for object detection, navigation, etc.
        """
        # Example: Detect edges for navigation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Example: Find contours (could be used for obstacle detection)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # In a real implementation, this might:
        # - Detect obstacles
        # - Identify landmarks for navigation
        # - Recognize objects for manipulation
        # - Track human operators
        pass
    
    def analyze_pressure_distribution(self):
        """
        Analyze foot pressure data for balance assessment
        """
        left_total = sum(self.foot_pressures['left'])
        right_total = sum(self.foot_pressures['right'])
        
        total_support = left_total + right_total
        
        if total_support > 0:
            # Calculate center of pressure (CoP)
            # This is a simplified calculation - in reality, you'd use actual sensor positions
            left_cop_x = (self.foot_pressures['left'][0] + self.foot_pressures['left'][1] - 
                         self.foot_pressures['left'][2] - self.foot_pressures['left'][3]) / total_support
            left_cop_y = (self.foot_pressures['left'][0] + self.foot_pressures['left'][2] - 
                         self.foot_pressures['left'][1] - self.foot_pressures['left'][3]) / total_support
            
            right_cop_x = (self.foot_pressures['right'][0] + self.foot_pressures['right'][1] - 
                          self.foot_pressures['right'][2] - self.foot_pressures['right'][3]) / total_support
            right_cop_y = (self.foot_pressures['right'][0] + self.foot_pressures['right'][2] - 
                          self.foot_pressures['right'][1] - self.foot_pressures['right'][3]) / total_support
            
            # Store CoP data for balance analysis
            self.cop_data = {
                'left': np.array([left_cop_x, left_cop_y]),
                'right': np.array([right_cop_x, right_cop_y])
            }
    
    def balance_control_callback(self):
        """
        Main balance control loop
        """
        # Calculate CoM error (difference from desired position)
        desired_com_y = 0.0  # Keep CoM centered laterally
        desired_com_z = 0.85  # Maintain nominal height
        
        com_error = np.array([
            0.0,  # We don't control forward CoM position directly
            self.com_position[1] - desired_com_y  # Lateral error
        ])
        
        # Update PID error terms
        dt = 0.01  # 100 Hz control rate
        self.com_error_integral += com_error * dt
        self.com_error_derivative = (com_error - self.prev_com_error) / dt if dt > 0 else np.zeros(2)
        self.prev_com_error = com_error.copy()
        
        # Calculate balance correction command
        balance_cmd = Twist()
        
        # PID control for lateral balance
        balance_cmd.linear.y = (
            self.balance_pid_gains['kp'] * com_error[1] +
            self.balance_pid_gains['ki'] * self.com_error_integral[1] +
            self.balance_pid_gains['kd'] * self.com_error_derivative[1]
        )
        
        # Limit the correction command
        max_correction = 0.5  # m/s
        balance_cmd.linear.y = max(-max_correction, min(max_correction, balance_cmd.linear.y))
        
        # If CoM is significantly off, add angular correction
        if abs(com_error[1]) > 0.05:  # 5cm threshold
            balance_cmd.angular.z = -2.0 * com_error[1]  # Counteract lateral lean
        
        # Publish balance command
        self.balance_command_publisher.publish(balance_cmd)
        
        # Publish status
        status_msg = String()
        status_msg.data = f'CoM: ({self.com_position[0]:.3f}, {self.com_position[1]:.3f}, {self.com_position[2]:.3f}), ' \
                         f'Balance correction: {balance_cmd.linear.y:.3f}'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    
    subscriber_node = RobotSubscriberNode()
    
    try:
        rclpy.spin(subscriber_node)
    except KeyboardInterrupt:
        subscriber_node.get_logger().info('Interrupted by user')
    finally:
        subscriber_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services and actions

Services provide synchronous request/reply communication, while actions provide goal-oriented communication with feedback and status updates.

### Services Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from std_srvs.srv import SetBool, Trigger
from sensor_msgs.srv import SetCameraInfo
from humanoid_msgs.srv import SetJointImpedance, ExecuteMotionSequence
import threading
import time

class HumanoidServiceServer(Node):
    def __init__(self):
        super().__init__('humanoid_service_server')
        
        # Create callback groups for concurrent service handling
        self.motion_cb_group = MutuallyExclusiveCallbackGroup()
        self.safety_cb_group = MutuallyExclusiveCallbackGroup()
        self.config_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Motion services
        self.stand_service = self.create_service(
            Trigger, 
            '/humanoid/stand_up', 
            self.stand_callback,
            callback_group=self.motion_cb_group
        )
        
        self.sit_service = self.create_service(
            Trigger,
            '/humanoid/sit_down',
            self.sit_callback,
            callback_group=self.motion_cb_group
        )
        
        self.walk_service = self.create_service(
            Trigger,
            '/humanoid/start_walking',
            self.walk_callback,
            callback_group=self.motion_cb_group
        )
        
        self.stop_service = self.create_service(
            Trigger,
            '/humanoid/stop_walking',
            self.stop_callback,
            callback_group=self.motion_cb_group
        )
        
        # Safety services
        self.emergency_stop_service = self.create_service(
            SetBool,
            '/humanoid/emergency_stop',
            self.emergency_stop_callback,
            callback_group=self.safety_cb_group
        )
        
        self.balance_enable_service = self.create_service(
            SetBool,
            '/humanoid/balance_enable',
            self.balance_enable_callback,
            callback_group=self.safety_cb_group
        )
        
        # Configuration services
        self.set_impedance_service = self.create_service(
            SetJointImpedance,
            '/humanoid/set_joint_impedance',
            self.set_impedance_callback,
            callback_group=self.config_cb_group
        )
        
        self.execute_motion_service = self.create_service(
            ExecuteMotionSequence,
            '/humanoid/execute_motion',
            self.execute_motion_callback,
            callback_group=self.motion_cb_group
        )
        
        # Robot state
        self.is_balanced = True
        self.is_walking = False
        self.is_sitting = False
        self.joint_impedances = {}  # Current joint impedances
        
        # Thread for motion execution
        self.motion_execution_lock = threading.Lock()
        self.current_motion_thread = None
        
        self.get_logger().info('Humanoid Service Server initialized')
    
    def stand_callback(self, request, response):
        """
        Stand up service callback
        """
        self.get_logger().info('Received stand up request')
        
        if self.is_sitting:
            # Execute standing motion
            success = self.execute_stand_motion()
            if success:
                response.success = True
                response.message = 'Successfully stood up'
                self.is_sitting = False
            else:
                response.success = False
                response.message = 'Failed to stand up'
        else:
            response.success = False
            response.message = 'Robot is not sitting'
        
        return response
    
    def sit_callback(self, request, response):
        """
        Sit down service callback
        """
        self.get_logger().info('Received sit down request')
        
        if not self.is_sitting and self.is_balanced:
            # Execute sitting motion
            success = self.execute_sit_motion()
            if success:
                response.success = True
                response.message = 'Successfully sat down'
                self.is_sitting = True
            else:
                response.success = False
                response.message = 'Failed to sit down'
        else:
            response.success = False
            response.message = 'Robot cannot sit (not balanced or already sitting)'
        
        return response
    
    def walk_callback(self, request, response):
        """
        Start walking service callback
        """
        self.get_logger().info('Received start walking request')
        
        if not self.is_walking and not self.is_sitting:
            # Start walking
            success = self.start_walking()
            if success:
                response.success = True
                response.message = 'Successfully started walking'
                self.is_walking = True
            else:
                response.success = False
                response.message = 'Failed to start walking'
        else:
            response.success = False
            response.message = 'Cannot start walking (already walking or sitting)'
        
        return response
    
    def stop_callback(self, request, response):
        """
        Stop walking service callback
        """
        self.get_logger().info('Received stop walking request')
        
        if self.is_walking:
            # Stop walking
            success = self.stop_walking()
            if success:
                response.success = True
                response.message = 'Successfully stopped walking'
                self.is_walking = False
            else:
                response.success = False
                response.message = 'Failed to stop walking'
        else:
            response.success = False
            response.message = 'Not currently walking'
        
        return response
    
    def emergency_stop_callback(self, request, response):
        """
        Emergency stop service callback
        """
        self.get_logger().info('Emergency stop requested')
        
        if request.data:  # If emergency stop is activated
            # Stop all motion immediately
            self.emergency_stop()
            response.success = True
            response.message = 'Emergency stop activated'
        else:  # If emergency stop is deactivated
            # Resume operations (with caution)
            response.success = self.resume_operations()
            response.message = 'Operations resumed' if response.success else 'Resume failed - safety check required'
        
        return response
    
    def balance_enable_callback(self, request, response):
        """
        Enable/disable balance control service callback
        """
        self.get_logger().info(f'Balance control {"enabled" if request.data else "disabled"}')
        
        if request.data:
            self.enable_balance_control()
            response.success = True
            response.message = 'Balance control enabled'
        else:
            self.disable_balance_control()
            response.success = True
            response.message = 'Balance control disabled'
        
        return response
    
    def set_impedance_callback(self, request, response):
        """
        Set joint impedance service callback
        """
        self.get_logger().info(f'Setting joint impedances for joints: {request.joint_names}')
        
        try:
            # Validate inputs
            if len(request.joint_names) != len(request.stiffness_values) or \
               len(request.joint_names) != len(request.damping_values):
                response.success = False
                response.message = 'Mismatched array lengths for joint names and values'
                return response
            
            # Update joint impedances
            for i, joint_name in enumerate(request.joint_names):
                self.joint_impedances[joint_name] = {
                    'stiffness': request.stiffness_values[i],
                    'damping': request.damping_values[i]
                }
            
            # Apply impedances to robot
            success = self.apply_joint_impedances(request.joint_names, 
                                                request.stiffness_values, 
                                                request.damping_values)
            
            if success:
                response.success = True
                response.message = f'Successfully set impedances for {len(request.joint_names)} joints'
            else:
                response.success = False
                response.message = 'Failed to apply joint impedances'
        
        except Exception as e:
            self.get_logger().error(f'Error in set_impedance_callback: {str(e)}')
            response.success = False
            response.message = f'Error setting impedances: {str(e)}'
        
        return response
    
    def execute_motion_callback(self, request, response):
        """
        Execute motion sequence service callback
        """
        self.get_logger().info(f'Received motion sequence with {len(request.motions)} motions')
        
        # Check if robot is in a safe state to execute motion
        if self.is_sitting or (self.is_walking and request.allow_interrupt == False):
            response.success = False
            response.message = 'Cannot execute motion - robot is in unsafe state'
            return response
        
        # Execute motion sequence in a separate thread to not block service
        motion_thread = threading.Thread(
            target=self._execute_motion_sequence_thread,
            args=(request, response)
        )
        
        with self.motion_execution_lock:
            if self.current_motion_thread and self.current_motion_thread.is_alive():
                response.success = False
                response.message = 'Motion already in progress'
                return response
            
            self.current_motion_thread = motion_thread
            motion_thread.start()
        
        # Return immediately with success (motion will continue asynchronously)
        response.success = True
        response.message = f'Motion sequence started with {len(request.motions)} motions'
        return response
    
    def _execute_motion_sequence_thread(self, request, response):
        """
        Threaded function to execute motion sequence
        """
        try:
            for i, motion in enumerate(request.motions):
                self.get_logger().info(f'Executing motion {i+1}/{len(request.motions)}: {motion.name}')
                
                # Execute individual motion
                success = self.execute_single_motion(motion)
                
                if not success:
                    with self.motion_execution_lock:
                        self.current_motion_thread = None
                    response.success = False
                    response.message = f'Motion {i+1} ({motion.name}) failed'
                    return
                
                # Check for preemption
                if request.allow_interrupt and self.check_for_preemption():
                    self.get_logger().info('Motion sequence interrupted')
                    with self.motion_execution_lock:
                        self.current_motion_thread = None
                    response.success = False
                    response.message = 'Motion sequence interrupted'
                    return
            
            # Motion sequence completed successfully
            with self.motion_execution_lock:
                self.current_motion_thread = None
            response.success = True
            response.message = f'Successfully executed {len(request.motions)} motions'
        
        except Exception as e:
            self.get_logger().error(f'Error executing motion sequence: {str(e)}')
            with self.motion_execution_lock:
                self.current_motion_thread = None
            response.success = False
            response.message = f'Error executing motion sequence: {str(e)}'
    
    def check_for_preemption(self):
        """
        Check if current motion should be preempted
        """
        # In a real implementation, this might check for emergency stops,
        # balance issues, or other high-priority events
        return False
    
    def execute_stand_motion(self):
        """
        Execute standing motion sequence
        """
        # This would contain the actual motion sequence
        # For simulation, we'll just sleep and return success
        time.sleep(2.0)  # Simulate motion time
        return True
    
    def execute_sit_motion(self):
        """
        Execute sitting motion sequence
        """
        # This would contain the actual motion sequence
        time.sleep(2.0)  # Simulate motion time
        return True
    
    def start_walking(self):
        """
        Start walking motion
        """
        # This would initialize walking controller
        return True
    
    def stop_walking(self):
        """
        Stop walking motion
        """
        # This would stop walking controller
        return True
    
    def emergency_stop(self):
        """
        Execute emergency stop procedure
        """
        # Stop all robot motion
        # Enable brakes if available
        # Log the event
        self.get_logger().warn('EMERGENCY STOP - All motion halted')
        return True
    
    def resume_operations(self):
        """
        Resume operations after emergency stop
        """
        # Perform safety checks before resuming
        # Clear emergency stop flag
        return True
    
    def enable_balance_control(self):
        """
        Enable balance control system
        """
        self.is_balanced = True
        self.get_logger().info('Balance control enabled')
        return True
    
    def disable_balance_control(self):
        """
        Disable balance control system
        """
        self.is_balanced = False
        self.get_logger().info('Balance control disabled')
        return True
    
    def apply_joint_impedances(self, joint_names, stiffness_values, damping_values):
        """
        Apply joint impedance values to robot hardware
        """
        # This would interface with the robot's control system
        # to set the specified impedance parameters
        return True
    
    def execute_single_motion(self, motion):
        """
        Execute a single motion primitive
        """
        # This would execute the specific motion
        # based on the motion parameters
        time.sleep(0.5)  # Simulate execution time
        return True

def main(args=None):
    rclpy.init(args=args)
    
    service_server = HumanoidServiceServer()
    
    # Use multi-threaded executor to handle concurrent service calls
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(service_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        service_server.get_logger().info('Interrupted by user')
    finally:
        service_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Actions Implementation

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from humanoid_msgs.action import WalkToPose, ManipulateObject, ExecuteDance
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64
import time
import threading
from tf_transformations import quaternion_from_euler, euler_from_quaternion

class HumanoidActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_action_server')
        
        # Action servers
        self.walk_to_pose_server = ActionServer(
            self,
            WalkToPose,
            'walk_to_pose',
            self.execute_walk_to_pose,
            goal_callback=self.walk_to_pose_goal_callback,
            cancel_callback=self.walk_to_pose_cancel_callback
        )
        
        self.manipulate_object_server = ActionServer(
            self,
            ManipulateObject,
            'manipulate_object',
            self.execute_manipulate_object,
            goal_callback=self.manipulate_object_goal_callback,
            cancel_callback=self.manipulate_object_cancel_callback
        )
        
        self.execute_dance_server = ActionServer(
            self,
            ExecuteDance,
            'execute_dance',
            self.execute_dance,
            goal_callback=self.dance_goal_callback,
            cancel_callback=self.dance_cancel_callback
        )
        
        # Robot state
        self.current_pose = Pose()
        self.is_moving = False
        self.is_manipulating = False
        self.is_dancing = False
        
        # Active goals
        self.active_walk_goal = None
        self.active_manipulation_goal = None
        self.active_dance_goal = None
        
        self.get_logger().info('Humanoid Action Server initialized')
    
    def walk_to_pose_goal_callback(self, goal_request):
        """
        Handle incoming walk to pose goal
        """
        self.get_logger().info(
            f'Received walk to pose goal: ({goal_request.target_pose.pose.position.x}, '
            f'{goal_request.target_pose.pose.position.y}, '
            f'{goal_request.target_pose.pose.position.z})'
        )
        
        # Check if robot can accept the goal
        if self.is_manipulating or self.is_dancing:
            self.get_logger().warn('Rejecting walk goal - robot busy with other task')
            return GoalResponse.REJECT
        
        if self.is_moving:
            # Check if new goal is significantly different from current
            current_goal = self.active_walk_goal
            if current_goal:
                dist_to_new = self.calculate_distance(
                    self.current_pose.position, 
                    goal_request.target_pose.pose.position
                )
                
                if dist_to_new < 0.1:  # Less than 10cm difference
                    self.get_logger().info('New goal is too close to current - accepting but will finish current first')
                    return GoalResponse.ACCEPT
        
        return GoalResponse.ACCEPT
    
    def walk_to_pose_cancel_callback(self, goal_handle):
        """
        Handle cancellation of walk to pose goal
        """
        self.get_logger().info('Received cancel request for walk to pose')
        return CancelResponse.ACCEPT
    
    def execute_walk_to_pose(self, goal_handle):
        """
        Execute walk to pose action
        """
        self.get_logger().info('Executing walk to pose action')
        
        # Store active goal
        self.active_walk_goal = goal_handle
        self.is_moving = True
        
        # Get goal parameters
        target_pose = goal_handle.request.target_pose.pose
        tolerance = goal_handle.request.tolerance if hasattr(goal_handle.request, 'tolerance') else 0.1
        max_time = goal_handle.request.timeout.sec if hasattr(goal_handle.request, 'timeout') else 30.0
        
        # Calculate distance to target
        distance_to_target = self.calculate_distance(
            self.current_pose.position,
            target_pose.position
        )
        
        # Initialize feedback
        feedback_msg = WalkToPose.Feedback()
        feedback_msg.current_pose = self.current_pose
        feedback_msg.distance_remaining = distance_to_target
        
        # Start walking
        start_time = time.time()
        current_distance = distance_to_target
        
        while current_distance > tolerance and time.time() - start_time < max_time:
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = WalkToPose.Result()
                result.success = False
                result.message = 'Goal was cancelled'
                self.is_moving = False
                self.active_walk_goal = None
                return result
            
            # Update robot position (simulation)
            self.update_robot_position_towards_target(target_pose, 0.05)  # 5cm per step
            
            # Recalculate distance
            current_distance = self.calculate_distance(
                self.current_pose.position,
                target_pose.position
            )
            
            # Publish feedback
            feedback_msg.current_pose = self.current_pose
            feedback_msg.distance_remaining = current_distance
            goal_handle.publish_feedback(feedback_msg)
            
            # Sleep to simulate walking
            time.sleep(0.1)
        
        # Check if we reached the target
        result = WalkToPose.Result()
        if current_distance <= tolerance:
            result.success = True
            result.message = f'Successfully walked to target pose in {time.time() - start_time:.2f} seconds'
            goal_handle.succeed()
        else:
            result.success = False
            result.message = 'Failed to reach target within timeout'
            goal_handle.abort()
        
        # Clean up
        self.is_moving = False
        self.active_walk_goal = None
        
        return result
    
    def manipulate_object_goal_callback(self, goal_request):
        """
        Handle incoming manipulate object goal
        """
        self.get_logger().info(f'Received manipulate object goal: {goal_request.object_id}')
        
        if self.is_moving or self.is_dancing:
            self.get_logger().warn('Rejecting manipulation goal - robot busy with other task')
            return GoalResponse.REJECT
        
        return GoalResponse.ACCEPT
    
    def manipulate_object_cancel_callback(self, goal_handle):
        """
        Handle cancellation of manipulate object goal
        """
        self.get_logger().info('Received cancel request for manipulate object')
        return CancelResponse.ACCEPT
    
    def execute_manipulate_object(self, goal_handle):
        """
        Execute manipulate object action
        """
        self.get_logger().info(f'Executing manipulation of object: {goal_request.object_id}')
        
        self.active_manipulation_goal = goal_handle
        self.is_manipulating = True
        
        # Initialize feedback
        feedback_msg = ManipulateObject.Feedback()
        feedback_msg.current_phase = 'approaching_object'
        feedback_msg.progress_percentage = 0.0
        
        # Example manipulation sequence
        phases = [
            ('approach', 2.0),      # 2 seconds to approach
            ('grasp', 1.5),         # 1.5 seconds to grasp
            ('lift', 1.0),          # 1 second to lift
            ('move', 3.0),          # 3 seconds to move
            ('place', 1.5),         # 1.5 seconds to place
            ('retreat', 1.0)        # 1 second to retreat
        ]
        
        for i, (phase, duration) in enumerate(phases):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = ManipulateObject.Result()
                result.success = False
                result.message = f'Manipulation cancelled during {phase} phase'
                self.is_manipulating = False
                self.active_manipulation_goal = None
                return result
            
            # Update feedback
            feedback_msg.current_phase = phase
            feedback_msg.progress_percentage = (i + 1) / len(phases) * 100.0
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate phase execution
            time.sleep(duration)
        
        # Complete manipulation
        result = ManipulateObject.Result()
        result.success = True
        result.message = f'Successfully manipulated object {goal_request.object_id}'
        goal_handle.succeed()
        
        self.is_manipulating = False
        self.active_manipulation_goal = None
        
        return result
    
    def dance_goal_callback(self, goal_request):
        """
        Handle incoming dance goal
        """
        self.get_logger().info(f'Received dance goal: {goal_request.dance_name}')
        
        if self.is_moving or self.is_manipulating:
            self.get_logger().warn('Rejecting dance goal - robot busy with other task')
            return GoalResponse.REJECT
        
        return GoalResponse.ACCEPT
    
    def dance_cancel_callback(self, goal_handle):
        """
        Handle cancellation of dance goal
        """
        self.get_logger().info('Received cancel request for dance')
        return CancelResponse.ACCEPT
    
    def execute_dance(self, goal_handle):
        """
        Execute dance action
        """
        self.get_logger().info(f'Executing dance: {goal_handle.request.dance_name}')
        
        self.active_dance_goal = goal_handle
        self.is_dancing = True
        
        # Initialize feedback
        feedback_msg = ExecuteDance.Feedback()
        feedback_msg.current_move = 'starting'
        feedback_msg.beat_count = 0
        feedback_msg.progress_percentage = 0.0
        
        # Get dance sequence
        dance_moves = self.get_dance_sequence(goal_handle.request.dance_name)
        
        total_beats = sum(move.get('beats', 1) for move in dance_moves)
        current_beat = 0
        
        for i, move in enumerate(dance_moves):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = ExecuteDance.Result()
                result.success = False
                result.message = f'Dance cancelled during move {i+1}'
                self.is_dancing = False
                self.active_dance_goal = None
                return result
            
            # Execute move
            move_name = move.get('name', f'move_{i}')
            beats = move.get('beats', 1)
            duration = beats * 0.5  # 0.5 seconds per beat
            
            feedback_msg.current_move = move_name
            feedback_msg.beat_count = current_beat
            feedback_msg.progress_percentage = (current_beat / total_beats) * 100.0
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate move execution
            time.sleep(duration)
            
            current_beat += beats
        
        # Complete dance
        result = ExecuteDance.Result()
        result.success = True
        result.message = f'Successfully completed dance: {goal_handle.request.dance_name}'
        goal_handle.succeed()
        
        self.is_dancing = False
        self.active_dance_goal = None
        
        return result
    
    def calculate_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two points
        """
        return ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)**0.5
    
    def update_robot_position_towards_target(self, target_pose, step_size):
        """
        Update robot position simulation towards target
        """
        direction = [
            target_pose.position.x - self.current_pose.position.x,
            target_pose.position.y - self.current_pose.position.y,
            target_pose.position.z - self.current_pose.position.z
        ]
        
        distance = (direction[0]**2 + direction[1]**2 + direction[2]**2)**0.5
        
        if distance > step_size:
            # Move one step towards target
            scale = step_size / distance
            self.current_pose.position.x += direction[0] * scale
            self.current_pose.position.y += direction[1] * scale
            self.current_pose.position.z += direction[2] * scale
        else:
            # Reached target
            self.current_pose.position = target_pose.position
            self.current_pose.orientation = target_pose.orientation
    
    def get_dance_sequence(self, dance_name):
        """
        Get predefined dance sequence
        """
        sequences = {
            'hello': [
                {'name': 'wave_right_arm', 'beats': 2},
                {'name': 'nod_head', 'beats': 1},
                {'name': 'step_side', 'beats': 1},
                {'name': 'return_to_center', 'beats': 1}
            ],
            'celebrate': [
                {'name': 'raise_arms', 'beats': 1},
                {'name': 'jump', 'beats': 1},
                {'name': 'spin', 'beats': 2},
                {'name': 'lower_arms', 'beats': 1}
            ],
            'think': [
                {'name': 'scratch_head', 'beats': 2},
                {'name': 'tilt_head', 'beats': 1},
                {'name': 'point_up', 'beats': 1},
                {'name': 'shrug_shoulders', 'beats': 2}
            ]
        }
        
        return sequences.get(dance_name, [])

def main(args=None):
    rclpy.init(args=args)
    
    action_server = HumanoidActionServer()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(action_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch files and parameters

Launch files allow you to start multiple nodes with a single command and configure their parameters.

### Launch File Structure

```python
# humanoid_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    robot_namespace = LaunchConfiguration('robot_namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file_path = LaunchConfiguration('config_file_path')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'robot_namespace',
            default_value='humanoid_robot',
            description='Robot namespace for multi-robot setups'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'config_file_path',
            default_value=[FindPackageShare('humanoid_robot_bringup'), '/config/humanoid_params.yaml'],
            description='Path to configuration file'
        ),
        
        # Log startup info
        LogInfo(
            msg=['Starting Humanoid Robot with namespace: ', robot_namespace]
        ),
        
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=robot_namespace,
            parameters=[
                config_file_path,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/joint_states', [robot_namespace, '/joint_states'])
            ]
        ),
        
        # Joint state broadcaster
        Node(
            package='joint_state_broadcaster',
            executable='joint_state_broadcaster',
            name='joint_state_broadcaster',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),
        
        # IMU sensor broadcaster
        Node(
            package='imu_sensor_broadcaster',
            executable='imu_sensor_broadcaster',
            name='imu_sensor_broadcaster',
            namespace=robot_namespace,
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),
        
        # Robot controller
        Node(
            package='humanoid_robot_controller',
            executable='humanoid_controller',
            name='humanoid_controller',
            namespace=robot_namespace,
            parameters=[
                config_file_path,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/cmd_vel', [robot_namespace, '/cmd_vel']),
                ('/joint_commands', [robot_namespace, '/joint_commands'])
            ]
        ),
        
        # Balance controller
        Node(
            package='humanoid_balance_controller',
            executable='balance_controller',
            name='balance_controller',
            namespace=robot_namespace,
            parameters=[
                config_file_path,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/imu/data', [robot_namespace, '/imu/data']),
                ('/center_of_mass', [robot_namespace, '/center_of_mass'])
            ]
        ),
        
        # Vision system
        Node(
            package='humanoid_vision_system',
            executable='vision_node',
            name='vision_node',
            namespace=robot_namespace,
            parameters=[
                config_file_path,
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', [robot_namespace, '/camera/image_raw']),
                ('/object_detections', [robot_namespace, '/object_detections'])
            ]
        ),
        
        # Navigation system
        Node(
            package='nav2_bringup',
            executable='nav2_bringup',
            name='nav2_stack',
            namespace=robot_namespace,
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_nav2_config'),
                    'config',
                    'nav2_params.yaml'
                ]),
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/tf', 'tf'),
                ('/tf_static', 'tf_static')
            ]
        )
    ])

# Advanced launch file with conditional nodes
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_advanced_launch_description():
    # Declare launch arguments
    use_simulation = LaunchConfiguration('use_simulation')
    use_camera = LaunchConfiguration('use_camera')
    use_lidar = LaunchConfiguration('use_lidar')
    robot_model = LaunchConfiguration('robot_model')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_simulation',
            default_value='false',
            description='Use simulation environment'
        ),
        DeclareLaunchArgument(
            'use_camera',
            default_value='true',
            description='Enable camera processing nodes'
        ),
        DeclareLaunchArgument(
            'use_lidar',
            default_value='true',
            description='Enable LiDAR processing nodes'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='unitree_g1',
            description='Robot model to load'
        ),
        
        # Simulation launch (if enabled)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'),
                '/launch/gazebo.launch.py'
            ]),
            condition=IfCondition(use_simulation)
        ),
        
        # Robot state publisher with robot-specific URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description': 
                    PathJoinSubstitution([
                        FindPackageShare('humanoid_description'),
                        'urdf',
                        [robot_model, '.urdf.xacro']
                    ])
                }
            ],
            condition=UnlessCondition(use_simulation)
        ),
        
        # Camera processing node (if enabled)
        Node(
            package='humanoid_vision_system',
            executable='camera_processor',
            name='camera_processor',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_vision_system'),
                    'config',
                    'camera_params.yaml'
                ])
            ],
            condition=IfCondition(use_camera)
        ),
        
        # LiDAR processing node (if enabled)
        Node(
            package='humanoid_perception',
            executable='lidar_processor',
            name='lidar_processor',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_perception'),
                    'config',
                    'lidar_params.yaml'
                ])
            ],
            condition=IfCondition(use_lidar)
        ),
        
        # Walking controller
        Node(
            package='humanoid_locomotion',
            executable='walking_controller',
            name='walking_controller',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_locomotion'),
                    'config',
                    [robot_model, '_walking_params.yaml']
                ])
            ]
        )
    ])
```

### Parameter Configuration

```yaml
# config/humanoid_params.yaml
humanoid_robot:
  ros__parameters:
    # General robot parameters
    robot_name: "unitree_g1"
    robot_mass: 34.0  # kg
    com_height: 0.78  # m
    max_velocity: 1.2  # m/s
    max_angular_velocity: 0.5  # rad/s
    use_sim_time: false
    
    # Joint control parameters
    joint_controller:
      control_frequency: 100  # Hz
      position_gain: 100.0
      velocity_gain: 10.0
      effort_limit: 100.0  # Nm
    
    # Balance control parameters
    balance_controller:
      zmp_tracking_gain: 50.0
      com_tracking_gain: 30.0
      ankle_impedance:
        stiffness: [1000.0, 1000.0, 500.0, 100.0, 100.0, 50.0]
        damping: [100.0, 100.0, 50.0, 10.0, 10.0, 5.0]
    
    # Walking parameters
    walking_controller:
      step_length: 0.30  # m
      step_width: 0.20  # m
      step_height: 0.08  # m
      step_duration: 0.65  # s
      double_support_ratio: 0.1  # 10% of step in double support
      nominal_com_height: 0.78  # m
      com_offset_x: 0.02  # m (slightly forward)
    
    # Vision parameters
    vision_system:
      detection_threshold: 0.5
      tracking_lifetime: 5.0  # s
      max_detection_range: 5.0  # m
      camera_fps: 30
      image_width: 640
      image_height: 480
    
    # Safety parameters
    safety_limits:
      max_joint_position_error: 0.2  # rad
      max_imu_angle: 0.5  # rad (~28 degrees)
      min_foot_pressure: 10.0  # N
      emergency_stop_timeout: 5.0  # s
    
    # Diagnostic parameters
    diagnostics:
      cpu_usage_threshold: 80.0  # %
      memory_usage_threshold: 80.0  # %
      temperature_threshold: 70.0  # Celsius
      battery_low_threshold: 20.0  # %

# Separate config for navigation
humanoid_navigation:
  ros__parameters:
    # Planner parameters
    planner:
      global_planner: "nav2_navfn_planner/NavfnPlanner"
      local_planner: "nav2_basic_local_planner/BasicLocalPlanner"
      planner_frequency: 1.0
      expected_planner_frequency: 1.0
    
    # Controller parameters
    controller:
      controller_frequency: 20.0
      min_x_velocity_threshold: 0.001
      min_y_velocity_threshold: 0.5
      min_theta_velocity_threshold: 0.001
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
    
    # Costmap parameters
    local_costmap:
      local_costmap:
        ros__parameters:
          update_frequency: 5.0
          publish_frequency: 2.0
          global_frame: "odom"
          robot_base_frame: "base_link"
          use_sim_time: false
          rolling_window: true
          width: 3
          height: 3
          resolution: 0.05
          robot_radius: 0.3
          plugins: ["voxel_layer", "inflation_layer"]
          
          voxel_layer:
            plugin: "nav2_costmap_2d::VoxelLayer"
            enabled: True
            publish_voxel_map: True
            origin_z: 0.0
            z_resolution: 0.2
            z_voxels: 16
            max_obstacle_height: 2.0
            mark_threshold: 0
            observation_sources: "scan"
            scan:
              topic: "/scan"
              max_obstacle_height: 2.0
              clearing: True
              marking: True
              data_type: "LaserScan"
              raytrace_max_range: 3.0
              raytrace_min_range: 0.0
              obstacle_max_range: 2.5
              obstacle_min_range: 0.0
          
          inflation_layer:
            plugin: "nav2_costmap_2d::InflationLayer"
            enabled: True
            cost_scaling_factor: 3.0
            inflation_radius: 0.55
            inflate_unknown: False