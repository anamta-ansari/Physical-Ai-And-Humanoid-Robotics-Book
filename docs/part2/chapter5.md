---
title: ROS 2 Development with Python
sidebar_position: 2
description: Learn ROS 2 development with Python, including creating packages, publishers, subscribers, launch files, and debugging techniques
---

# ROS 2 Development with Python

## Creating ROS 2 packages

ROS 2 packages are the basic building blocks of ROS 2 applications. They contain nodes, libraries, and other resources needed for specific functionality.

### Package Structure

A typical ROS 2 Python package has the following structure:

```
my_robot_package/
├── package.xml          # Package manifest
├── CMakeLists.txt       # Build configuration (for mixed C++/Python)
├── setup.py             # Python setup configuration
├── setup.cfg            # Installation configuration
├── my_robot_package/    # Python module directory
│   ├── __init__.py      # Python package initialization
│   ├── my_node.py       # Python node implementation
│   └── my_module.py     # Additional Python modules
└── launch/              # Launch files directory
    └── my_launch_file.launch.py
```

### Creating a Package

To create a new ROS 2 package for Python development:

```bash
ros2 pkg create --build-type ament_python my_robot_package
```

This command creates the basic package structure with necessary configuration files.

### Package.xml Configuration

The `package.xml` file contains metadata about the package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Package for my robot functionality</description>
  <maintainer email="user@example.com">User Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup Configuration

The `setup.py` file configures how the Python package is built and installed:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Package for my robot functionality',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = my_robot_package.my_node:main',
        ],
    },
)
```

## Writing publishers and subscribers

Publishers and subscribers are fundamental to ROS 2 communication using the publish/subscribe pattern.

### Creating a Publisher

Here's an example of a simple publisher node in Python:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Subscriber

Here's an example of a subscriber node in Python:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service Settings

Quality of Service (QoS) settings allow you to configure message delivery characteristics:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,  # or BEST_EFFORT
    history=HistoryPolicy.KEEP_LAST,  # or KEEP_ALL
)

# Use the QoS profile when creating publisher/subscriber
publisher = self.create_publisher(String, 'topic', qos_profile)
subscriber = self.create_subscription(String, 'topic', callback, qos_profile)
```

## Launch files

Launch files allow you to start multiple nodes with a single command and configure their parameters.

### Basic Launch File

Here's an example of a simple launch file:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_node',
            parameters=[
                {'param1': 'value1'},
                {'param2': 123},
            ],
            remappings=[
                ('original_topic', 'remapped_topic'),
            ],
            arguments=['arg1', 'arg2'],
        ),
        Node(
            package='another_package',
            executable='another_node',
            name='another_node',
        ),
    ])
```

### Advanced Launch File with Conditions

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    debug = LaunchConfiguration('debug')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'),
        DeclareLaunchArgument(
            'debug',
            default_value='false',
            description='Enable debug mode if true'),
        
        # Node that runs only in debug mode
        Node(
            package='my_robot_package',
            executable='debug_node',
            name='debug_node',
            condition=IfCondition(debug),
        ),
        
        # Node with parameters
        Node(
            package='my_robot_package',
            executable='main_node',
            name='main_node',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_name': 'my_robot'},
            ],
        ),
    ])
```

### Launch File Commands

To run a launch file:

```bash
ros2 launch my_robot_package my_launch_file.launch.py
```

With arguments:

```bash
ros2 launch my_robot_package my_launch_file.launch.py use_sim_time:=true debug:=true
```

## Parameters and configurations

Parameters allow nodes to be configured at runtime without recompilation.

### Using Parameters in Nodes

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('topics_to_subscribe', ['topic1', 'topic2'])
        
        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.topics = self.get_parameter('topics_to_subscribe').value
        
        # Set a parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)
    
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.get_logger().info(f'Updated max velocity to: {param.value}')
                self.max_velocity = param.value
        return SetParametersResult(successful=True)
```

### Parameter Files

Parameters can be loaded from YAML files:

```yaml
# config/my_robot_params.yaml
parameter_node:
  ros__parameters:
    robot_name: 'my_custom_robot'
    max_velocity: 2.0
    topics_to_subscribe: ['sensor_data', 'control_commands']
```

Loading parameters from a file:

```python
import os
from ament_index_python.packages import get_package_share_directory

# In the launch file
Node(
    package='my_robot_package',
    executable='parameter_node',
    name='parameter_node',
    parameters=[
        os.path.join(get_package_share_directory('my_robot_package'), 'config', 'my_robot_params.yaml')
    ],
)
```

## rclpy for robot control

rclpy is the Python client library for ROS 2 that provides the interface to ROS 2 functionality.

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Create publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Create subscribers for sensor data
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Create timers for control loops
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        # Initialize robot state
        self.current_position = None
        self.target_position = None
        
    def odom_callback(self, msg):
        self.current_position = msg.pose.pose
        
    def control_loop(self):
        # Implement control logic here
        if self.target_position and self.current_position:
            cmd = self.calculate_control_command()
            self.cmd_vel_publisher.publish(cmd)
```

### Working with Time and Timers

```python
from rclpy.time import Time
from rclpy.duration import Duration

class TimedNode(Node):
    def __init__(self):
        super().__init__('timed_node')
        
        # Create a timer
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Record start time
        self.start_time = self.get_clock().now()
    
    def timer_callback(self):
        current_time = self.get_clock().now()
        elapsed = current_time - self.start_time
        
        self.get_logger().info(f'Elapsed time: {elapsed.nanoseconds / 1e9:.2f}s')
```

### Services and Actions in Python

```python
from rclpy.action import ActionClient
from rclpy.service import Service
from example_interfaces.action import Fibonacci
from example_interfaces.srv import AddTwoInts

class ServiceActionNode(Node):
    def __init__(self):
        super().__init__('service_action_node')
        
        # Create a service
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_callback)
        
        # Create an action client
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')
    
    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response
    
    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback.sequence
        self.get_logger().info(f'Received feedback: {feedback}')
```

## Debugging ROS 2 applications

Debugging ROS 2 applications requires understanding of both Python debugging techniques and ROS 2-specific tools.

### Using Python Debuggers

```python
import rclpy
import pdb  # Python debugger

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    
    # Set a breakpoint
    pdb.set_trace()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### ROS 2 Debugging Tools

```bash
# Check node status
ros2 node info <node_name>

# Monitor topics
ros2 topic echo <topic_name>
ros2 topic info <topic_name>

# Monitor services
ros2 service list
ros2 service info <service_name>

# Monitor actions
ros2 action list
ros2 action info <action_name>

# Parameter inspection
ros2 param list
ros2 param get <node_name> <param_name>
```

### Logging in ROS 2

```python
class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')
        
        # Different logging levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')
        
        # Formatted logging
        value = 42
        self.get_logger().info(f'Value is: {value}')
```

## Conclusion

ROS 2 development with Python provides a powerful framework for creating robotic applications. Understanding how to create packages, implement publishers and subscribers, use launch files, manage parameters, and debug applications is essential for effective robot development. The rclpy library provides all the necessary tools to build sophisticated robotic systems, from simple sensor processing to complex control algorithms for humanoid robots.

## Next Steps

To continue learning about ROS 2 development:

- Review [Chapter 4: Introduction to ROS 2](../part2/chapter4) for fundamental concepts
- Continue to [Chapter 6: Robot Description (URDF & XACRO)](../part2/chapter6) to understand how to model robots for ROS