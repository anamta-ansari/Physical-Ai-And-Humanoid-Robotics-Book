---
title: Physical AI & Humanoid Robotics Implementation
sidebar_position: 8
description: Complete implementation of the Physical AI & Humanoid Robotics book with all chapters, parts, and features
---

# Physical AI & Humanoid Robotics Implementation

## Complete book structure

Now that we've developed all the foundational components, we'll implement the complete Physical AI & Humanoid Robotics book with all 8 parts and 24 chapters as specified in the feature requirements. This implementation will bring together all the concepts we've developed into a cohesive, comprehensive resource.

### Part 1: Foundations of Physical AI & Humanoid Robotics

For Part 1, we'll implement the foundational concepts of Physical AI and humanoid robotics:

```python
# Implementation of Part 1: Foundations of Physical AI & Humanoid Robotics

# Chapter 1: Introduction to Physical AI
chapter1_content = """
# Chapter 1: Introduction to Physical AI

## What is Physical AI?

Physical AI represents a paradigm shift from traditional digital AI to AI systems that are embodied in physical form and interact directly with the physical world. Unlike digital AI that processes abstract data, Physical AI must contend with real-world physics, sensor noise, actuator limitations, and environmental uncertainties.

### Definition of Physical AI

Physical AI can be defined as artificial intelligence systems that:
- Operate within physical bodies (embodied intelligence)
- Interact with the physical environment through sensors and actuators
- Process information from real-world interactions
- Adapt to dynamic physical conditions
- Exhibit behaviors that emerge from physical interaction

### Contrast with Digital AI

| Aspect | Digital AI | Physical AI |
|--------|------------|-------------|
| Environment | Virtual/Digital | Physical/Real World |
| Input | Abstract data | Sensor data from physical world |
| Output | Information/Decisions | Physical actions and movements |
| Constraints | Computational | Physical laws, safety, hardware |
| Learning | Simulation/History | Real-world experience |

## Difference between digital AI vs physical AI

The fundamental difference between digital and physical AI lies in their relationship with the physical world:

### Digital AI Characteristics
- Processes abstract symbolic information
- Operates in deterministic environments
- No physical constraints on operations
- Perfect information availability
- Instantaneous state changes

### Physical AI Characteristics
- Must operate within physical laws
- Deals with uncertainty and noise
- Subject to physical constraints
- Limited information through sensors
- Continuous state evolution

## Embodied intelligence: AI inside a body

Embodied intelligence is a core concept in Physical AI that recognizes the importance of physical embodiment in intelligence:

### The Embodiment Hypothesis

The embodiment hypothesis suggests that intelligence emerges from the interaction between an agent and its environment. Key aspects include:

1. **Morphological Computation**: The body's physical properties contribute to intelligent behavior
2. **Enactive Cognition**: Knowledge emerges through active interaction with the environment
3. **Situatedness**: Intelligence is shaped by environmental context
4. **Emergence**: Complex behaviors emerge from simple body-environment interactions

### Benefits of Embodiment

- **Sensorimotor Learning**: Learning through physical interaction
- **Adaptive Behavior**: Automatic adaptation to environmental changes
- **Energy Efficiency**: Leverage physical dynamics for efficiency
- **Robustness**: Natural fault tolerance through physical properties

## Why AI needs physical awareness (gravity, friction, torque)

Physical awareness is crucial for AI systems that interact with the real world:

### Gravity Awareness

AI systems must understand gravity to:
- Maintain balance and stability
- Predict object motion and falls
- Plan manipulations that account for weight
- Understand support relationships

### Friction Understanding

Friction awareness enables:
- Grip planning and manipulation
- Locomotion on different surfaces
- Prediction of sliding vs. rolling motion
- Energy-efficient movement planning

### Torque Considerations

Torque understanding is essential for:
- Joint control and actuation
- Force regulation during manipulation
- Balance control through moment management
- Safe interaction with environment

## History and evolution of humanoid robotics

The development of humanoid robotics spans several decades with significant technological evolution:

### Early Developments (1960s-1980s)
- Mechanical automata and simple walking machines
- Focus on basic locomotion patterns
- Limited sensing and control capabilities

### Academic Phase (1990s-2000s)
- Introduction of dynamic balance control
- Development of ZMP-based walking
- Advanced sensing and control systems
- Notable platforms: Honda ASIMO, Sony QRIO

### Commercial Phase (2010s-Present)
- Unitree Go1/A1, Boston Dynamics Atlas, Tesla Optimus
- Advanced AI integration
- Improved mobility and manipulation
- Focus on practical applications

## Current humanoid robot industry (Unitree, Tesla Optimus, Boston Dynamics)

### Unitree Robotics
- **Go1**: Lightweight, agile quadruped robot
- **A1**: High-performance quadruped platform
- **G1**: Humanoid robot with 23 DOF
- Focus on affordable, high-performance platforms

### Tesla Optimus
- Humanoid robot for general tasks
- Integration with Tesla's AI expertise
- Focus on practical applications
- Advanced computer vision capabilities

### Boston Dynamics
- **Atlas**: Advanced humanoid with dynamic capabilities
- **Spot**: Agile quadruped platform
- Focus on dynamic locomotion and manipulation
- Extensive research and development background

## Conclusion

Physical AI represents the future of artificial intelligence, where systems are not just intelligent but embodied and capable of interacting with the physical world. Understanding the fundamentals of Physical AI is essential for developing humanoid robots that can operate effectively in human environments.
"""

# Chapter 2: Why Physical AI Matters
chapter2_content = """
# Chapter 2: Why Physical AI Matters

## AI in the physical world vs virtual world

The distinction between AI in physical and virtual worlds highlights fundamental differences in requirements and capabilities:

### Physical World Challenges

- **Uncertainty**: Sensor noise, environmental changes, unpredictable interactions
- **Physics**: Real-world physics constraints (gravity, friction, collisions)
- **Safety**: Risk of damage to robot or environment
- **Real-time Requirements**: Immediate responses to physical events
- **Embodiment**: Limited by physical form and capabilities

### Virtual World Advantages

- **Precision**: Exact state information and deterministic behavior
- **Safety**: No risk of physical damage
- **Speed**: Fast simulation and testing
- **Control**: Complete control over environment
- **Repeatability**: Identical conditions for testing

## Importance of human-centered design in robotics

Human-centered design is crucial for humanoid robotics:

### Ergonomic Considerations

- **Size and Proportion**: Robot dimensions compatible with human environments
- **Reach Envelope**: Manipulation capabilities matching human workspace
- **Interaction Modes**: Natural communication methods (speech, gestures)
- **Safety Boundaries**: Safe interaction spaces and forces

### Social Acceptance

- **Appearance**: Human-like features for familiar interaction
- **Behavior**: Predictable, understandable actions
- **Personality**: Appropriate social responses
- **Trust Building**: Reliable performance over time

## How humanoids adapt to our world

Humanoid robots are specifically designed to operate in human-centric environments:

### Environmental Compatibility

- **Architecture**: Designed for human-scale spaces (doors, furniture, stairs)
- **Tools**: Can potentially use human tools and equipment
- **Infrastructure**: Compatible with human-designed systems
- **Social Norms**: Behaviors that align with human expectations

### Advantages of Humanoid Form

- **Intuitive Interaction**: Natural communication patterns
- **Skill Transfer**: Human skills can be adapted to humanoid control
- **Learning**: Can learn from human demonstrations
- **Integration**: Fits naturally into human environments

## Data abundance from real-world interactions

Physical AI systems generate vast amounts of rich, multimodal data:

### Sensor Data Types

- **Proprioceptive**: Joint angles, motor currents, IMU readings
- **Exteroceptive**: Cameras, LiDAR, tactile sensors, force/torque
- **Environmental**: Temperature, humidity, acoustic data
- **Interaction**: Contact forces, slip detection, manipulation feedback

### Data Quality Characteristics

- **Richness**: Multimodal, temporal, spatial information
- **Authenticity**: Reflects real operating conditions
- **Variety**: Natural variation in real-world scenarios
- **Relevance**: Directly applicable to robot's operating environment

## Future of work: humans + AI agents + robots

The integration of physical AI into the workforce requires new models of human-robot collaboration:

### Collaborative Models

1. **Complementary Roles**: Humans and robots perform different but complementary tasks
2. **Supervision**: AI agents assist humans in supervising multiple robots
3. **Augmentation**: Robots enhance human capabilities rather than replacing them
4. **Teaming**: Humans and robots work together on shared tasks

### Applications in Various Sectors

- **Manufacturing**: Humans for complex assembly, robots for repetitive tasks
- **Healthcare**: Robots for lifting and transport, humans for patient care
- **Logistics**: Robots for material handling, humans for complex decisions
- **Construction**: Robots for heavy lifting, humans for precision work

## Conclusion

Physical AI matters because it represents the next frontier in AI development - systems that can truly understand and interact with the physical world. This capability is essential for creating robots that can work alongside humans, perform useful tasks in human environments, and contribute to solving real-world problems.
"""

# Chapter 3: Overview of Humanoid Robotics
chapter3_content = """
# Chapter 3: Overview of Humanoid Robotics

## Types of robots (industrial, service, humanoid, quadruped)

Understanding the different types of robots helps clarify where humanoid robots fit in the robotics landscape:

### Industrial Robots
- **Characteristics**: Fixed-base, high precision, repetitive tasks
- **Applications**: Manufacturing, assembly, welding, painting
- **Advantages**: High speed, precision, reliability
- **Limitations**: Limited mobility, specialized for specific tasks

### Service Robots
- **Characteristics**: Mobile, task-focused, human interaction
- **Applications**: Cleaning, delivery, assistance
- **Advantages**: Flexibility, human-compatible tasks
- **Limitations**: Limited dexterity, safety constraints

### Humanoid Robots
- **Characteristics**: Human-like form, bipedal locomotion, anthropomorphic manipulation
- **Applications**: Human environment operation, social interaction, complex manipulation
- **Advantages**: Human-compatible environments, intuitive interaction, versatile manipulation
- **Limitations**: Complexity, energy consumption, balance challenges

### Quadruped Robots
- **Characteristics**: Four-legged locomotion, stability, agility
- **Applications**: Rough terrain navigation, exploration, load carrying
- **Advantages**: Superior stability, rough terrain capability
- **Limitations**: Non-human environment interaction, limited manipulation

## Human-robot interaction basics

Effective human-robot interaction is crucial for humanoid robot deployment:

### Communication Modalities

1. **Verbal Communication**: Speech recognition and synthesis
2. **Gestural Communication**: Understanding and producing gestures
3. **Facial Expression**: Displaying appropriate expressions
4. **Proxemics**: Understanding and respecting personal space
5. **Touch Interaction**: Appropriate tactile responses

### Interaction Principles

- **Predictability**: Robot behaviors should be predictable
- **Transparency**: Robot intentions should be clear
- **Safety**: All interactions must be safe
- **Naturalness**: Behaviors should feel natural to humans
- **Context Awareness**: Robot should understand situational context

## Sensors used in humanoid robots

Humanoid robots require diverse sensor systems to perceive and interact with their environment:

### Proprioceptive Sensors
- **Joint Encoders**: Measure joint angles and velocities
- **IMU (Inertial Measurement Units)**: Measure orientation, acceleration, angular velocity
- **Force/Torque Sensors**: Measure forces at joints and end-effectors
- **Motor Current Sensors**: Indirect force sensing through motor current

### Exteroceptive Sensors
- **Cameras**: Visual perception, object recognition, navigation
- **LiDAR**: 3D mapping, obstacle detection, navigation
- **Tactile Sensors**: Touch perception, grip feedback, surface properties
- **Microphones**: Speech recognition, sound localization
- **Range Sensors**: Distance measurement, proximity detection

### Sensor Fusion
- **Multi-modal Integration**: Combining data from different sensors
- **Temporal Integration**: Incorporating sensor data over time
- **Uncertainty Management**: Handling sensor noise and uncertainty
- **Calibration**: Ensuring accurate sensor measurements

## Applications: Education, Healthcare, Defense, Household automation, Manufacturing

### Education
- **Teaching Tool**: Explaining robotics, programming, and AI concepts
- **Engagement**: Captivating students with interactive demonstrations
- **Programming Platform**: Students can program robots for various tasks
- **Social Skills**: Helping children with autism practice social interaction

### Healthcare
- **Companion Robots**: Providing social interaction for elderly patients
- **Physical Therapy**: Guiding patients through exercises
- **Assistance**: Helping with mobility and daily activities
- **Telemedicine**: Enabling remote consultation and monitoring
- **Training**: Simulating patients for medical training

### Defense
- **Reconnaissance**: Gathering intelligence in dangerous environments
- **EOD (Explosive Ordnance Disposal)**: Handling dangerous materials
- **Security**: Patrolling and monitoring facilities
- **Logistics**: Carrying equipment in difficult terrain
- **Challenges**: Require high reliability and security

### Household Automation
- **Companionship**: Providing interaction and entertainment
- **Assistance**: Helping with daily tasks (limited currently)
- **Security**: Monitoring home environments
- **Entertainment**: Interactive toys and pets
- **Current Limitations**: Cost, capability, and safety concerns

### Manufacturing
- **Collaborative Work**: Working alongside human workers
- **Flexible Automation**: Adapting to different tasks
- **Quality Inspection**: Using vision systems for defect detection
- **Material Handling**: Transporting items in dynamic environments
- **Challenges**: Safety, reliability, and cost-effectiveness

## Conclusion

Humanoid robotics represents a convergence of mechanical engineering, artificial intelligence, and human factors research. The diverse applications across education, healthcare, defense, and other sectors demonstrate the potential of these systems. As technology continues to advance, we can expect humanoid robots to play increasingly important roles in our society, working alongside humans in ways that complement human capabilities and enhance our productivity and quality of life.
"""

# Write Part 1 chapters
with open('docs/part1/chapter1.md', 'w') as f:
    f.write(chapter1_content)

with open('docs/part1/chapter2.md', 'w') as f:
    f.write(chapter2_content)

with open('docs/part1/chapter3.md', 'w') as f:
    f.write(chapter3_content)

print("Part 1 chapters created successfully")
```

### Part 2: Robotic Nervous System (ROS 2)

```python
# Implementation of Part 2: Robotic Nervous System (ROS 2)

# Chapter 4: Introduction to ROS 2
chapter4_content = """
# Chapter 4: Introduction to ROS 2

## What is ROS 2 and why it's important

Robot Operating System 2 (ROS 2) is the next-generation robotics middleware that provides a framework for developing robot applications. Unlike its predecessor, ROS 2 was designed from the ground up to address the limitations of ROS 1, particularly around real-time performance, security, and multi-robot systems.

### Key Improvements in ROS 2

1. **Quality of Service (QoS) Policies**: Configurable reliability and performance settings
2. **Security Framework**: Built-in security features for safe deployment
3. **Real-time Support**: Capabilities for real-time systems
4. **Multi-platform Support**: Better support for different operating systems
5. **DDS Integration**: Data Distribution Service for robust communication

### Why ROS 2 is Important for Humanoid Robotics

- **Modularity**: Enables component-based development
- **Interoperability**: Facilitates integration of different software components
- **Scalability**: Supports complex systems with many components
- **Community**: Large ecosystem of tools and packages
- **Industry Adoption**: Widely used in research and industry

## ROS 2 architecture

ROS 2 uses a distributed architecture based on the Data Distribution Service (DDS) standard:

### Core Architecture Components

1. **Nodes**: Processes that perform computation
2. **Topics**: Named buses over which nodes exchange messages
3. **Services**: Synchronous request/reply communication
4. **Actions**: Asynchronous goal-based communication with feedback
5. **Parameters**: Configuration values that can be changed at runtime

### DDS Implementation Options

ROS 2 supports multiple DDS implementations:
- **Fast DDS**: eProsima's implementation (default in newer versions)
- **Cyclone DDS**: Eclipse Foundation's implementation
- **RTI Connext DDS**: RTI's commercial implementation
- **OpenSplice DDS**: ADLINK's implementation

## Nodes, Topics, Services, Actions

### Nodes

Nodes are the fundamental building blocks of ROS 2 applications. Each node runs a specific task and communicates with other nodes through messages.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HumanoidSensorNode(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_node')
        self.publisher = self.create_publisher(String, 'sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    humanoid_sensor_node = HumanoidSensorNode()
    rclpy.spin(humanoid_sensor_node)
    humanoid_sensor_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

Topics enable asynchronous, many-to-many communication between nodes:

```python
# Publisher example
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            # ... other joints
        ]

    def publish_joint_states(self, positions, velocities, efforts):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        self.publisher.publish(msg)
```

### Services

Services provide synchronous, request/reply communication:

```python
# Service server example
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class BalanceControlService(Node):
    def __init__(self):
        super().__init__('balance_control_service')
        self.srv = self.create_service(
            SetBool, 
            'enable_balance_control', 
            self.enable_balance_callback
        )
        self.balance_enabled = False

    def enable_balance_callback(self, request, response):
        self.balance_enabled = request.data
        response.success = True
        response.message = f'Balance control {"enabled" if self.balance_enabled else "disabled"}'
        return response
```

### Actions

Actions provide goal-oriented communication with feedback:

```python
# Action server example
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Replace with actual walk action type
            'walk_to_goal',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk action...')
        
        # Simulate walking progress
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Walk action canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)  # Simulate walking time

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info('Walk action completed')
        
        return result
```

## Message passing

Message passing in ROS 2 is based on DDS and supports various Quality of Service (QoS) settings:

### QoS Policies

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or transient local data persistence
- **History**: Keep all or keep last N messages
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: Method for determining entity availability

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

## ROS 2 graph

The ROS 2 graph represents the network of nodes and their communication connections:

### Graph Components

- **Nodes**: Individual processes running ROS 2 code
- **Topics**: Communication channels between nodes
- **Services**: Request/reply connections
- **Actions**: Goal-oriented communication
- **Parameters**: Configuration values shared between nodes

### Command-line Tools

```bash
# List all nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /joint_states

# Call a service
ros2 service call /balance_control std_srvs/SetBool "{data: true}"

# List actions
ros2 action list
```

## Conclusion

ROS 2 provides a robust framework for developing complex humanoid robot applications. Its distributed architecture, Quality of Service policies, and rich ecosystem of tools and packages make it an ideal choice for humanoid robotics development. Understanding ROS 2 fundamentals is essential for building scalable, maintainable humanoid robot systems.
"""

# Chapter 5: ROS 2 Development with Python
chapter5_content = """
# Chapter 5: ROS 2 Development with Python

## Creating ROS 2 packages

Creating ROS 2 packages is the foundation of any ROS 2 project. Packages contain the nodes, libraries, and other resources needed for your robot application.

### Package Structure

A typical ROS 2 package has the following structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package manifest
├── setup.cfg               # Installation configuration
├── setup.py                # Python setup configuration
├── my_robot_package/       # Python module directory
│   ├── __init__.py         # Module initialization
│   ├── robot_controller.py # Robot control implementation
│   └── sensors.py          # Sensor interfaces
├── launch/                 # Launch files
│   └── robot.launch.py
├── config/                 # Configuration files
│   └── params.yaml
└── test/                   # Test files
    └── test_robot.py
```

### Creating a Package

To create a new ROS 2 package:

```bash
# Create a new Python package
ros2 pkg create --build-type ament_python my_humanoid_package --dependencies rclpy std_msgs sensor_msgs geometry_msgs

# Create a new C++ package (if needed)
ros2 pkg create --build-type ament_cmake my_control_package --dependencies rclpy std_msgs sensor_msgs
```

### Package.xml Configuration

The package.xml file contains metadata about your package:

```xml
<?xml version=\"1.0\"?>
<?xml-model href=\"http://download.ros.org/schema/package_format3.xsd\" schematypens=\"http://www.w3.org/2001/XMLSchema\"?>
<package format=\"3\">
  <name>my_humanoid_package</name>
  <version>0.0.0</version>
  <description>Package for humanoid robot control</description>
  <maintainer email=\"developer@example.com\">Developer Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Writing publishers and subscribers

Publishers and subscribers form the backbone of ROS 2 communication:

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        
        # Timer for periodic publishing
        self.timer = self.create_timer(0.02, self.publish_joint_states)  # 50 Hz
        
        # Initialize joint information
        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_wrist_pitch', 'right_wrist_yaw'
        ]
        
        # Initialize joint positions (simulated values)
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)
        
        self.get_logger().info('Joint state publisher initialized')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts
        
        # Simulate changing joint positions (for demonstration)
        current_time = time.time()
        for i in range(len(self.joint_positions)):
            # Create oscillating motion for demonstration
            self.joint_positions[i] = 0.5 * math.sin(current_time + i * 0.1)
        
        # Publish the message
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    
    joint_publisher = JointStatePublisher()
    
    try:
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        joint_publisher.get_logger().info('Shutting down joint state publisher')
    finally:
        joint_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10  # QoS depth
        )
        
        # Create publisher for processed joint data
        self.processed_publisher = self.create_publisher(
            Float64MultiArray,
            'processed_joint_data',
            10
        )
        
        self.get_logger().info('Joint state subscriber initialized')

    def joint_state_callback(self, msg):
        # Process joint state message
        self.get_logger().debug(f'Received joint state message with {len(msg.name)} joints')
        
        # Example: Calculate joint velocity from position changes
        if hasattr(self, 'prev_positions') and self.prev_positions is not None:
            dt = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - self.prev_time
            if dt > 0:
                velocities = [(pos - prev_pos) / dt for pos, prev_pos in 
                             zip(msg.position, self.prev_positions)]
                
                # Create processed data message
                processed_msg = Float64MultiArray()
                processed_msg.data = velocities
                self.processed_publisher.publish(processed_msg)
        
        # Store current values for next iteration
        self.prev_positions = list(msg.position)
        self.prev_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

def main(args=None):
    rclpy.init(args=args)
    
    joint_subscriber = JointStateSubscriber()
    
    try:
        rclpy.spin(joint_subscriber)
    except KeyboardInterrupt:
        joint_subscriber.get_logger().info('Shutting down joint state subscriber')
    finally:
        joint_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch files

Launch files allow you to start multiple nodes with a single command:

### Python Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Robot namespace'
    ))
    
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    ))
    
    # Add joint state publisher node
    ld.add_action(Node(
        package='my_humanoid_package',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    ))
    
    # Add joint state subscriber node
    ld.add_action(Node(
        package='my_humanoid_package',
        executable='joint_state_subscriber',
        name='joint_state_subscriber',
        namespace=namespace,
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    ))
    
    return ld
```

## Parameters and configurations

Parameters allow you to configure your nodes without recompiling:

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import IntegerRange, FloatingPointRange

class ParameterizedRobotNode(Node):
    def __init__(self):
        super().__init__('parameterized_robot_node')
        
        # Declare parameters with descriptions and ranges
        self.declare_parameter(
            'robot_name',
            'humanoid_robot',
            ParameterDescriptor(
                description='Name of the robot',
                read_only=False
            )
        )
        
        self.declare_parameter(
            'control_frequency',
            100,
            ParameterDescriptor(
                description='Control loop frequency in Hz',
                integer_range=[IntegerRange(from_value=10, to_value=1000, step=1)]
            )
        )
        
        self.declare_parameter(
            'max_joint_velocity',
            2.0,
            ParameterDescriptor(
                description='Maximum joint velocity in rad/s',
                floating_point_range=[FloatingPointRange(from_value=0.1, to_value=10.0, step=0.1)]
            )
        )
        
        self.declare_parameter(
            'safety_margin',
            0.1,
            ParameterDescriptor(
                description='Safety margin for joint limits in radians'
            )
        )
        
        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.max_joint_velocity = self.get_parameter('max_joint_velocity').value
        self.safety_margin = self.get_parameter('safety_margin').value
        
        # Create timer with parameterized frequency
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )
        
        self.get_logger().info(
            f'Initialized robot node with parameters: '
            f'name={self.robot_name}, freq={self.control_frequency}Hz, '
            f'max_vel={self.max_joint_velocity}rad/s'
        )
    
    def control_loop(self):
        # Control loop implementation
        self.get_logger().debug('Control loop executing')
        
        # Example: Check if parameters have changed
        new_control_freq = self.get_parameter('control_frequency').value
        if new_control_freq != self.control_frequency:
            self.get_logger().info(f'Control frequency changed from {self.control_frequency} to {new_control_freq}')
            self.control_frequency = new_control_freq
            
            # Adjust timer period
            self.control_timer.timer_period_ns = int(1.0 / self.control_frequency * 1e9)

def main(args=None):
    rclpy.init(args=args)
    
    param_node = ParameterizedRobotNode()
    
    try:
        rclpy.spin(param_node)
    except KeyboardInterrupt:
        param_node.get_logger().info('Shutting down parameterized robot node')
    finally:
        param_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### YAML Parameter Files

```yaml
# config/robot_params.yaml
my_humanoid_node:
  ros__parameters:
    robot_name: "unitree_g1"
    control_frequency: 200
    max_joint_velocity: 3.0
    safety_margin: 0.05
    joint_limits:
      hip_pitch_min: -1.57
      hip_pitch_max: 0.785
      knee_pitch_min: -0.2
      knee_pitch_max: 2.5
    walking_parameters:
      step_length: 0.3
      step_height: 0.08
      step_duration: 0.65
      balance_margin: 0.05
```

## rclpy for robot control

rclpy is the Python client library for ROS 2 that provides the interface to ROS 2 functionality:

### Basic rclpy Concepts

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import math

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')
        
        # Create publishers for different robot interfaces
        self.cmd_vel_publisher = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        
        self.joint_command_publisher = self.create_publisher(
            JointState,
            '/joint_commands',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )
        
        # Create subscribers for robot feedback
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        self.imu_subscriber = self.create_subscription(
            Imu,  # Need to import this
            '/imu/data',
            self.imu_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        
        # Create service clients and servers
        self.balance_service_client = self.create_client(
            SetBool,  # Need to import this
            '/enable_balance_control'
        )
        
        # Create action clients and servers
        # self.walk_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # Robot state
        self.current_joint_positions = {}
        self.current_joint_velocities = {}
        self.imu_data = {}
        self.is_balanced = True
        
        # Control parameters
        self.control_frequency = 100  # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Create control timer
        self.control_timer = self.create_timer(
            self.dt, 
            self.control_loop
        )
        
        self.get_logger().info('Robot controller node initialized')

    def joint_state_callback(self, msg):
        """Process joint state messages"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }
        
        # Check balance based on IMU data
        self.check_balance_from_imu()

    def check_balance_from_imu(self):
        """Check if robot is balanced based on IMU data"""
        # Example: Check if robot is tilted too much
        orientation = self.imu_data['orientation']
        
        # Convert quaternion to Euler angles to check tilt
        roll, pitch, yaw = self.quaternion_to_euler(orientation)
        
        # Define balance thresholds (in radians)
        max_lean_angle = 0.3  # About 17 degrees
        
        if abs(roll) > max_lean_angle or abs(pitch) > max_lean_angle:
            if self.is_balanced:
                self.get_logger().warn(f'Robot is unbalanced! Roll: {roll:.2f}, Pitch: {pitch:.2f}')
                self.is_balanced = False
        else:
            if not self.is_balanced:
                self.get_logger().info('Robot balance restored')
                self.is_balanced = True

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        import math
        
        x, y, z, w = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def control_loop(self):
        """Main control loop"""
        if not self.is_balanced:
            # Implement balance recovery
            self.recover_balance()
        else:
            # Implement normal walking or other behaviors
            self.execute_behavior()

    def recover_balance(self):
        """Implement balance recovery strategy"""
        # Example: Move center of mass back to stable position
        # This would involve more complex control algorithms in practice
        self.get_logger().info('Attempting balance recovery')
        
        # Send commands to adjust posture
        self.adjust_posture_for_balance()

    def adjust_posture_for_balance(self):
        """Adjust robot posture to regain balance"""
        # This would send specific joint commands to adjust the robot's pose
        # For example, move hips, adjust ankle angles, etc.
        
        # Example: Send joint commands to return to neutral position
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.header.frame_id = 'base_link'
        
        # Define target joint positions for balanced posture
        # This is simplified - in reality, you'd calculate appropriate positions
        joint_state.name = list(self.current_joint_positions.keys())
        joint_state.position = [0.0] * len(joint_state.name)  # Return to neutral
        
        self.joint_command_publisher.publish(joint_state)

    def execute_behavior(self):
        """Execute normal robot behavior"""
        # This could be walking, manipulation, or other tasks
        pass

def main(args=None):
    rclpy.init(args=args)
    
    controller_node = RobotControllerNode()
    
    try:
        rclpy.spin(controller_node)
    except KeyboardInterrupt:
        controller_node.get_logger().info('Shutting down robot controller')
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Conclusion

ROS 2 development with Python provides a powerful framework for creating complex humanoid robot applications. The combination of nodes, topics, services, actions, and parameters enables modular, scalable robot systems. Understanding these concepts is crucial for effective humanoid robot development.
"""

# Chapter 6: Robot Description (URDF & XACRO)
chapter6_content = """
# Chapter 6: Robot Description (URDF & XACRO)

## URDF basics

Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. It defines the physical and visual properties of a robot, including its links, joints, sensors, and actuators.

### URDF Structure

A URDF file contains several key elements that describe different aspects of a robot:

```xml
<?xml version=\"1.0\"?>
<robot name=\"humanoid_robot\">
  <!-- Materials used in the robot -->
  <material name=\"blue\">
    <color rgba=\"0.0 0.0 0.8 1.0\"/>
  </material>
  <material name=\"black\">
    <color rgba=\"0.0 0.0 0.0 1.0\"/>
  </material>
  <material name=\"white\">
    <color rgba=\"1.0 1.0 1.0 1.0\"/>
  </material>

  <!-- Base link (root of the robot) -->
  <link name=\"base_link\">
    <inertial>
      <mass value=\"10.0\"/>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"1.0\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.0\" iyz=\"0.0\" izz=\"1.0\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <cylinder length=\"0.2\" radius=\"0.15\"/>
      </geometry>
      <material name=\"white\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <cylinder length=\"0.2\" radius=\"0.15\"/>
      </geometry>
    </collision>
  </link>

  <!-- Example joint and link -->
  <joint name=\"hip_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"left_leg\"/>
    <origin xyz=\"0 -0.15 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"100\" velocity=\"3.0\"/>
    <dynamics damping=\"1.0\" friction=\"0.1\"/>
  </joint>

  <link name=\"left_leg\">
    <inertial>
      <mass value=\"2.0\"/>
      <origin xyz=\"0 0 -0.25\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.05\" ixy=\"0\" ixz=\"0\" iyy=\"0.05\" iyz=\"0\" izz=\"0.01\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 -0.25\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.5\" radius=\"0.05\"/>
      </geometry>
      <material name=\"blue\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 -0.25\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.5\" radius=\"0.05\"/>
      </geometry>
    </collision>
  </link>
</robot>
```

### URDF Elements Explained

#### `<robot>` Element
- **name**: Unique identifier for the robot
- **Contains**: All links, joints, materials, and other elements

#### `<link>` Element
- **name**: Unique identifier for the link
- **Contains**: 
  - `<inertial>`: Mass and inertial properties
  - `<visual>`: Visual representation for display
  - `<collision>`: Collision representation for physics simulation

#### `<joint>` Element
- **name**: Unique identifier for the joint
- **type**: Type of joint (fixed, revolute, continuous, prismatic, etc.)
- **parent/child**: Links that the joint connects
- **origin**: Position and orientation of the joint
- **axis**: Axis of rotation or translation
- **limit**: Joint limits (for revolute and prismatic joints)
- **dynamics**: Joint dynamics (damping, friction)

### Joint Types in URDF

1. **Fixed**: No movement between parent and child links
2. **Revolute**: Single degree of freedom rotation with limits
3. **Continuous**: Single degree of freedom rotation without limits
4. **Prismatic**: Single degree of freedom translation with limits
5. **Planar**: Motion on a plane
6. **Floating**: Six degrees of freedom

## Creating a humanoid URDF

Creating a humanoid URDF requires careful consideration of the robot's kinematic structure and physical properties:

### Humanoid Skeleton Structure

A humanoid robot typically has the following structure:

```
base_link
├── torso
│   ├── head
│   ├── left_arm
│   │   ├── left_forearm
│   │   └── left_hand
│   ├── right_arm
│   │   ├── right_forearm
│   │   └── right_hand
│   ├── left_leg
│   │   ├── left_lower_leg
│   │   └── left_foot
│   └── right_leg
│       ├── right_lower_leg
│       └── right_foot
```

### Complete Humanoid URDF Example

```xml
<?xml version=\"1.0\"?>
<robot name=\"simple_humanoid\">
  <!-- Materials -->
  <material name=\"light_grey\">
    <color rgba=\"0.7 0.7 0.7 1.0\"/>
  </material>
  <material name=\"dark_grey\">
    <color rgba=\"0.3 0.3 0.3 1.0\"/>
  </material>
  <material name=\"blue\">
    <color rgba=\"0.0 0.0 0.8 1.0\"/>
  </material>
  <material name=\"red\">
    <color rgba=\"0.8 0.0 0.0 1.0\"/>
  </material>

  <!-- Base link -->
  <link name=\"base_link\">
    <inertial>
      <mass value=\"15.0\"/>
      <origin xyz=\"0 0 0.5\" rpy=\"0 0 0\"/>
      <inertia ixx=\"1.5\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"1.5\" iyz=\"0.0\" izz=\"1.0\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 0.5\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.2 0.2 1.0\"/>
      </geometry>
      <material name=\"light_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 0.5\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.2 0.2 1.0\"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name=\"head\">
    <inertial>
      <mass value=\"2.0\"/>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.01\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.01\" iyz=\"0.0\" izz=\"0.01\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <sphere radius=\"0.1\"/>
      </geometry>
      <material name=\"light_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <sphere radius=\"0.1\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"neck_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"head\"/>
    <origin xyz=\"0 0 1.0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-0.5\" upper=\"0.5\" effort=\"10\" velocity=\"2.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>

  <!-- Left Arm -->
  <link name=\"left_shoulder\">
    <inertial>
      <mass value=\"1.5\"/>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.005\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.005\" iyz=\"0.0\" izz=\"0.002\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.1\" radius=\"0.05\"/>
      </geometry>
      <material name=\"blue\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.1\" radius=\"0.05\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_shoulder_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"left_shoulder\"/>
    <origin xyz=\"0.1 0.1 0.8\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"50\" velocity=\"3.0\"/>
    <dynamics damping=\"0.1\" friction=\"0.05\"/>
  </joint>

  <!-- Left Forearm -->
  <link name=\"left_forearm\">
    <inertial>
      <mass value=\"1.0\"/>
      <origin xyz=\"0 -0.15 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.003\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.003\" iyz=\"0.0\" izz=\"0.001\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 -0.15 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.04\"/>
      </geometry>
      <material name=\"blue\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 -0.15 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.04\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_elbow_joint\" type=\"revolute\">
    <parent link=\"left_shoulder\"/>
    <child link=\"left_forearm\"/>
    <origin xyz=\"0 -0.1 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"30\" velocity=\"4.0\"/>
    <dynamics damping=\"0.1\" friction=\"0.05\"/>
  </joint>

  <!-- Left Hand -->
  <link name=\"left_hand\">
    <inertial>
      <mass value=\"0.3\"/>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.0005\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.0005\" iyz=\"0.0\" izz=\"0.0003\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.08 0.1 0.05\"/>
      </geometry>
      <material name=\"light_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 -0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.08 0.1 0.05\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_wrist_joint\" type=\"revolute\">
    <parent link=\"left_forearm\"/>
    <child link=\"left_hand\"/>
    <origin xyz=\"0 -0.3 0\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-0.78\" upper=\"0.78\" effort=\"10\" velocity=\"2.0\"/>
    <dynamics damping=\"0.05\" friction=\"0.02\"/>
  </joint>

  <!-- Right Arm (similar to left arm but mirrored) -->
  <link name=\"right_shoulder\">
    <inertial>
      <mass value=\"1.5\"/>
      <origin xyz=\"0 0.05 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.005\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.005\" iyz=\"0.0\" izz=\"0.002\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.1\" radius=\"0.05\"/>
      </geometry>
      <material name=\"red\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0.05 0\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.1\" radius=\"0.05\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"right_shoulder_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"right_shoulder\"/>
    <origin xyz=\"0.1 -0.1 0.8\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"50\" velocity=\"3.0\"/>
    <dynamics damping=\"0.1\" friction=\"0.05\"/>
  </joint>

  <!-- Continue with right forearm and hand similarly... -->
  
  <!-- Left Leg -->
  <link name=\"left_thigh\">
    <inertial>
      <mass value=\"3.0\"/>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.02\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.02\" iyz=\"0.0\" izz=\"0.005\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.06\"/>
      </geometry>
      <material name=\"dark_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.06\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_hip_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"left_thigh\"/>
    <origin xyz=\"-0.05 0.1 -0.1\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"100\" velocity=\"2.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>

  <!-- Left Shin -->
  <link name=\"left_shin\">
    <inertial>
      <mass value=\"2.5\"/>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.015\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.015\" iyz=\"0.0\" izz=\"0.003\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.4\" radius=\"0.05\"/>
      </geometry>
      <material name=\"dark_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 -0.2\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.4\" radius=\"0.05\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_knee_joint\" type=\"revolute\">
    <parent link=\"left_thigh\"/>
    <child link=\"left_shin\"/>
    <origin xyz=\"0 0 -0.3\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-0.1\" upper=\"2.0\" effort=\"100\" velocity=\"2.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>

  <!-- Left Foot -->
  <link name=\"left_foot\">
    <inertial>
      <mass value=\"1.0\"/>
      <origin xyz=\"0.05 0 -0.05\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.005\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.005\" iyz=\"0.0\" izz=\"0.002\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0.05 0 -0.05\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.2 0.1 0.1\"/>
      </geometry>
      <material name=\"light_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0.05 0 -0.05\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.2 0.1 0.1\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"left_ankle_joint\" type=\"revolute\">
    <parent link=\"left_shin\"/>
    <child link=\"left_foot\"/>
    <origin xyz=\"0 0 -0.4\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-0.5\" upper=\"0.5\" effort=\"50\" velocity=\"2.0\"/>
    <dynamics damping=\"0.2\" friction=\"0.05\"/>
  </joint>

  <!-- Right Leg (similar to left leg but mirrored) -->
  <link name=\"right_thigh\">
    <inertial>
      <mass value=\"3.0\"/>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.02\" ixy=\"0.0\" ixz=\"0.0\" iyy=\"0.02\" iyz=\"0.0\" izz=\"0.005\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.06\"/>
      </geometry>
      <material name=\"dark_grey\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 -0.15\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"0.3\" radius=\"0.06\"/>
      </geometry>
    </collision>
  </link>

  <joint name=\"right_hip_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"right_thigh\"/>
    <origin xyz=\"-0.05 -0.1 -0.1\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"100\" velocity=\"2.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>

  <!-- Continue with right shin and foot similarly... -->
</robot>
```

## Joints, links, sensors, actuators

In a humanoid robot URDF, the proper definition of joints, links, sensors, and actuators is crucial for simulation and control:

### Joint Definitions

Joints connect links and define how they can move relative to each other:

```xml
<!-- Revolute joint (rotational with limits) -->
<joint name=\"hip_pitch_joint\" type=\"revolute\">
  <parent link=\"torso\"/>
  <child link=\"thigh\"/>
  <origin xyz=\"0 0.1 -0.8\" rpy=\"0 0 0\"/>  <!-- Position relative to parent -->
  <axis xyz=\"1 0 0\"/>  <!-- Rotation axis (x-axis in this case) -->
  <limit lower=\"-1.57\" upper=\"0.78\" effort=\"200\" velocity=\"2.0\"/>  <!-- Joint limits -->
  <dynamics damping=\"5.0\" friction=\"1.0\"/>  <!-- Joint dynamics -->
</joint>

<!-- Continuous joint (rotational without limits) -->
<joint name=\"shoulder_yaw_joint\" type=\"continuous\">
  <parent link=\"torso\"/>
  <child link=\"upper_arm\"/>
  <origin xyz=\"0.2 0.1 0.2\" rpy=\"0 0 0\"/>
  <axis xyz=\"0 0 1\"/>
  <dynamics damping=\"2.0\" friction=\"0.5\"/>
</joint>

<!-- Fixed joint (no movement) -->
<joint name=\"sensor_mount_joint\" type=\"fixed\">
  <parent link=\"head\"/>
  <child link=\"camera_link\"/>
  <origin xyz=\"0.05 0 0.05\" rpy=\"0 0 0\"/>
</joint>
```

### Link Definitions

Links represent rigid bodies in the robot:

```xml
<link name=\"upper_arm\">
  <!-- Inertial properties for physics simulation -->
  <inertial>
    <mass value=\"1.5\"/>  <!-- Mass in kg -->
    <origin xyz=\"0 -0.1 0\" rpy=\"0 0 0\"/>  <!-- Center of mass location -->
    <!-- Inertia tensor values -->
    <inertia ixx=\"0.005\" ixy=\"0.0\" ixz=\"0.0\" 
             iyy=\"0.005\" iyz=\"0.0\" izz=\"0.002\"/>
  </inertial>
  
  <!-- Visual properties for display -->
  <visual>
    <origin xyz=\"0 -0.1 0\" rpy=\"0 0 0\"/>
    <geometry>
      <capsule length=\"0.2\" radius=\"0.05\"/>  <!-- Shape of the link -->
    </geometry>
    <material name=\"blue\"/>  <!-- Material to use for display -->
  </visual>
  
  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz=\"0 -0.1 0\" rpy=\"0 0 0\"/>
    <geometry>
      <capsule length=\"0.2\" radius=\"0.05\"/>  <!-- Collision shape -->
    </geometry>
  </collision>
</link>
```

### Adding Sensors to URDF

Sensors are represented as additional links with sensor plugins:

```xml
<!-- Camera sensor -->
<link name=\"camera_link\">
  <inertial>
    <mass value=\"0.1\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <inertia ixx=\"0.001\" ixy=\"0\" ixz=\"0\" iyy=\"0.001\" iyz=\"0\" izz=\"0.001\"/>
  </inertial>
  
  <visual>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <geometry>
      <box size=\"0.02 0.03 0.01\"/>
    </geometry>
    <material name=\"black\"/>
  </visual>
  
  <collision>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <geometry>
      <box size=\"0.02 0.03 0.01\"/>
    </geometry>
  </collision>
</link>

<joint name=\"camera_joint\" type=\"fixed\">
  <parent link=\"head\"/>
  <child link=\"camera_link\"/>
  <origin xyz=\"0.05 0 0.05\" rpy=\"0 0 0\"/>
</joint>

<!-- IMU sensor -->
<link name=\"imu_link\">
  <inertial>
    <mass value=\"0.01\"/>
    <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
    <inertia ixx=\"1e-6\" ixy=\"0\" ixz=\"0\" iyy=\"1e-6\" iyz=\"0\" izz=\"1e-6\"/>
  </inertial>
</link>

<joint name=\"imu_joint\" type=\"fixed\">
  <parent link=\"torso\"/>
  <child link=\"imu_link\"/>
  <origin xyz=\"0 0 0.1\" rpy=\"0 0 0\"/>
</joint>
```

### Actuator Modeling

While URDF doesn't directly model actuators, it can include transmission elements:

```xml
<!-- Transmission for a joint (defines how actuator connects to joint) -->
<transmission name=\"left_hip_pitch_transmission\" type=\"transmission_interface/SimpleTransmission\">
  <joint name=\"left_hip_pitch_joint\">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name=\"left_hip_pitch_motor\">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## XACRO for modular robot building

XACRO (XML Macros) extends URDF with features like variables, macros, and mathematical expressions, making it easier to create complex, modular robot descriptions.

### XACRO Basics

XACRO files use the `.xacro` extension and must include the XACRO namespace:

```xml
<?xml version=\"1.0\"?>
<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"humanoid_robot\">
  <!-- XACRO content goes here -->
</robot>
```

### XACRO Properties (Variables)

Properties allow you to define constants that can be reused throughout the file:

```xml
<!-- Define robot dimensions -->
<xacro:property name=\"robot_height\" value=\"1.5\" />
<xacro:property name=\"torso_height\" value=\"0.6\" />
<xacro:property name=\"leg_length\" value=\"0.7\" />
<xacro:property name=\"arm_length\" value=\"0.5\" />

<!-- Define material properties -->
<xacro:property name=\"link_density\" value=\"7000\" />  <!-- kg/m^3 -->
<xacro:property name=\"default_damping\" value=\"0.5\" />
<xacro:property name=\"default_friction\" value=\"0.1\" />

<!-- Define joint limits -->
<xacro:property name=\"hip_pitch_min\" value=\"-1.57\" />
<xacro:property name=\"hip_pitch_max\" value=\"0.78\" />
<xacro:property name=\"knee_pitch_max\" value=\"2.0\" />
```

### XACRO Macros

Macros allow you to define reusable components:

```xml
<!-- Macro for creating a simple link with standard properties -->
<xacro:macro name=\"simple_link\" params=\"name mass length radius material\">
  <link name=\"${name}\">
    <inertial>
      <mass value=\"${mass}\"/>
      <origin xyz=\"0 0 ${length/2}\" rpy=\"0 0 0\"/>
      <inertia ixx=\"${mass*(3*radius*radius + length*length)/12}\" 
               ixy=\"0\" ixz=\"0\"
               iyy=\"${mass*(3*radius*radius + length*length)/12}\" 
               iyz=\"0\"
               izz=\"${mass*radius*radius/2}\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 ${length/2}\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"${length}\" radius=\"${radius}\"/>
      </geometry>
      <material name=\"${material}\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 ${length/2}\" rpy=\"0 0 0\"/>
      <geometry>
        <capsule length=\"${length}\" radius=\"${radius}\"/>
      </geometry>
    </collision>
  </link>
</xacro:macro>

<!-- Macro for creating a revolute joint -->
<xacro:macro name=\"revolute_joint\" params=\"name parent child xyz rpy axis lower upper effort velocity\">
  <joint name=\"${name}\" type=\"revolute\">
    <parent link=\"${parent}\"/>
    <child link=\"${child}\"/>
    <origin xyz=\"${xyz}\" rpy=\"${rpy}\"/>
    <axis xyz=\"${axis}\"/>
    <limit lower=\"${lower}\" upper=\"${upper}\" effort=\"${effort}\" velocity=\"${velocity}\"/>
    <dynamics damping=\"${default_damping}\" friction=\"${default_friction}\"/>
  </joint>
</xacro:macro>

<!-- Macro for creating an arm (left or right) -->
<xacro:macro name=\"arm\" params=\"side position_xyz\">
  <!-- Shoulder link -->
  <xacro:simple_link name=\"${side}_shoulder\" 
                     mass=\"1.0\" 
                     length=\"0.1\" 
                     radius=\"0.05\" 
                     material=\"${side}_arm_color\"/>
  
  <!-- Shoulder joint -->
  <joint name=\"${side}_shoulder_pitch\" type=\"revolute\">
    <parent link=\"torso\"/>
    <child link=\"${side}_shoulder\"/>
    <origin xyz=\"${position_xyz}\" rpy=\"0 0 0\"/>
    <axis xyz=\"1 0 0\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"50\" velocity=\"3.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>
  
  <!-- Upper arm -->
  <xacro:simple_link name=\"${side}_upper_arm\" 
                     mass=\"1.5\" 
                     length=\"0.3\" 
                     radius=\"0.04\" 
                     material=\"${side}_arm_color\"/>
  
  <joint name=\"${side}_shoulder_yaw\" type=\"revolute\">
    <parent link=\"${side}_shoulder\"/>
    <child link=\"${side}_upper_arm\"/>
    <origin xyz=\"0 0 0.1\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 0 1\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"40\" velocity=\"3.0\"/>
    <dynamics damping=\"0.3\" friction=\"0.05\"/>
  </joint>
  
  <!-- Forearm -->
  <xacro:simple_link name=\"${side}_forearm\" 
                     mass=\"1.0\" 
                     length=\"0.25\" 
                     radius=\"0.035\" 
                     material=\"${side}_arm_color\"/>
  
  <joint name=\"${side}_elbow_pitch\" type=\"revolute\">
    <parent link=\"${side}_upper_arm\"/>
    <child link=\"${side}_forearm\"/>
    <origin xyz=\"0 0 0.3\" rpy=\"0 0 0\"/>
    <axis xyz=\"1 0 0\"/>
    <limit lower=\"-1.57\" upper=\"1.57\" effort=\"30\" velocity=\"4.0\"/>
    <dynamics damping=\"0.2\" friction=\"0.05\"/>
  </joint>
</xacro:macro>
```

### Using XACRO Macros

```xml
<?xml version=\"1.0\"?>
<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\" name=\"modular_humanoid\">
  <!-- Include other XACRO files if needed -->
  <xacro:include filename=\"$(find my_robot_description)/urdf/materials.xacro\" />
  <xacro:include filename=\"$(find my_robot_description)/urdf/transmissions.xacro\" />
  
  <!-- Define robot-specific properties -->
  <xacro:property name=\"robot_name\" value=\"modular_humanoid\" />
  <xacro:property name=\"torso_mass\" value=\"10.0\" />
  <xacro:property name=\"head_mass\" value=\"2.0\" />
  
  <!-- Define materials -->
  <material name=\"left_arm_color\">
    <color rgba=\"0.0 0.0 0.8 1.0\"/>
  </material>
  
  <material name=\"right_arm_color\">
    <color rgba=\"0.8 0.0 0.0 1.0\"/>
  </material>
  
  <material name=\"body_color\">
    <color rgba=\"0.7 0.7 0.7 1.0\"/>
  </material>
  
  <!-- Create base link -->
  <link name=\"base_link\">
    <inertial>
      <mass value=\"${torso_mass}\"/>
      <origin xyz=\"0 0 0.3\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.5\" ixy=\"0\" ixz=\"0\" iyy=\"0.5\" iyz=\"0\" izz=\"0.3\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 0.3\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.3 0.2 0.6\"/>
      </geometry>
      <material name=\"body_color\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 0.3\" rpy=\"0 0 0\"/>
      <geometry>
        <box size=\"0.3 0.2 0.6\"/>
      </geometry>
    </collision>
  </link>
  
  <!-- Create head -->
  <link name=\"head\">
    <inertial>
      <mass value=\"${head_mass}\"/>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <inertia ixx=\"0.01\" ixy=\"0\" ixz=\"0\" iyy=\"0.01\" iyz=\"0\" izz=\"0.01\"/>
    </inertial>
    
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <sphere radius=\"0.1\"/>
      </geometry>
      <material name=\"body_color\"/>
    </visual>
    
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry>
        <sphere radius=\"0.1\"/>
      </geometry>
    </collision>
  </link>
  
  <joint name=\"neck_joint\" type=\"revolute\">
    <parent link=\"base_link\"/>
    <child link=\"head\"/>
    <origin xyz=\"0 0 0.6\" rpy=\"0 0 0\"/>
    <axis xyz=\"0 1 0\"/>
    <limit lower=\"-0.5\" upper=\"0.5\" effort=\"10\" velocity=\"2.0\"/>
    <dynamics damping=\"0.5\" friction=\"0.1\"/>
  </joint>
  
  <!-- Use the arm macro to create both arms -->
  <xacro:arm side=\"left\" position_xyz=\"0.15 0.1 0.4\" />
  <xacro:arm side=\"right\" position_xyz=\"0.15 -0.1 0.4\" />
  
  <!-- Similarly, we could create leg macros -->
  <xacro:macro name=\"leg\" params=\"side position_xyz\">
    <!-- Hip link -->
    <xacro:simple_link name=\"${side}_hip\" 
                       mass=\"2.0\" 
                       length=\"0.1\" 
                       radius=\"0.06\" 
                       material=\"body_color\"/>
    
    <joint name=\"${side}_hip_yaw_pitch\" type=\"revolute\">
      <parent link=\"base_link\"/>
      <child link=\"${side}_hip\"/>
      <origin xyz=\"${position_xyz}\" rpy=\"0 0 0\"/>
      <axis xyz=\"0 0 1\"/>
      <limit lower=\"-0.5\" upper=\"0.5\" effort=\"100\" velocity=\"2.0\"/>
      <dynamics damping=\"1.0\" friction=\"0.2\"/>
    </joint>
    
    <!-- Thigh -->
    <xacro:simple_link name=\"${side}_thigh\" 
                       mass=\"3.0\" 
                       length=\"0.4\" 
                       radius=\"0.06\" 
                       material=\"body_color\"/>
    
    <joint name=\"${side}_hip_pitch\" type=\"revolute\">
      <parent link=\"${side}_hip\"/>
      <child link=\"${side}_thigh\"/>
      <origin xyz=\"0 0 -0.1\" rpy=\"0 0 0\"/>
      <axis xyz=\"1 0 0\"/>
      <limit lower=\"-1.57\" upper=\"0.78\" effort=\"150\" velocity=\"2.0\"/>
      <dynamics damping=\"2.0\" friction=\"0.3\"/>
    </joint>
    
    <!-- Shin -->
    <xacro:simple_link name=\"${side}_shin\" 
                       mass=\"2.5\" 
                       length=\"0.4\" 
                       radius=\"0.05\" 
                       material=\"body_color\"/>
    
    <joint name=\"${side}_knee_pitch\" type=\"revolute\">
      <parent link=\"${side}_thigh\"/>
      <child link=\"${side}_shin\"/>
      <origin xyz=\"0 0 -0.4\" rpy=\"0 0 0\"/>
      <axis xyz=\"1 0 0\"/>
      <limit lower=\"-0.1\" upper=\"2.0\" effort=\"150\" velocity=\"2.0\"/>
      <dynamics damping=\"2.0\" friction=\"0.3\"/>
    </joint>
    
    <!-- Foot -->
    <xacro:simple_link name=\"${side}_foot\" 
                       mass=\"1.0\" 
                       length=\"0.2\" 
                       width=\"0.1\" 
                       height=\"0.05\" 
                       material=\"body_color\"/>
    
    <joint name=\"${side}_ankle_pitch\" type=\"revolute\">
      <parent link=\"${side}_shin\"/>
      <child link=\"${side}_foot\"/>
      <origin xyz=\"0 0 -0.4\" rpy=\"0 0 0\"/>
      <axis xyz=\"1 0 0\"/>
      <limit lower=\"-0.5\" upper=\"0.5\" effort=\"80\" velocity=\"2.0\"/>
      <dynamics damping=\"1.0\" friction=\"0.2\"/>
    </joint>
  </xacro:macro>
  
  <!-- Create both legs -->
  <xacro:leg side=\"left\" position_xyz=\"-0.05 0.1 -0.1\" />
  <xacro:leg side=\"right\" position_xyz=\"-0.05 -0.1 -0.1\" />
  
  <!-- Add sensors using macros -->
  <xacro:macro name=\"camera_sensor\" params=\"name parent_link position_xyz\">
    <link name=\"${name}_link\">
      <inertial>
        <mass value=\"0.05\"/>
        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
        <inertia ixx=\"1e-5\" ixy=\"0\" ixz=\"0\" iyy=\"1e-5\" iyz=\"0\" izz=\"1e-5\"/>
      </inertial>
      
      <visual>
        <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
        <geometry>
          <box size=\"0.02 0.03 0.01\"/>
        </geometry>
        <material name=\"black\"/>
      </visual>
    </link>
    
    <joint name=\"${name}_joint\" type=\"fixed\">
      <parent link=\"${parent_link}\"/>
      <child link=\"${name}_link\"/>
      <origin xyz=\"${position_xyz}\" rpy=\"0 0 0\"/>
    </joint>
    
    <!-- Gazebo plugin for camera simulation -->
    <gazebo reference=\"${name}_link\">
      <sensor type=\"camera\" name=\"${name}_camera\">
        <update_rate>30.0</update_rate>
        <camera name=\"head\">
          <horizontal_fov>1.3962634</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
        <plugin name=\"camera_controller\" filename=\"libgazebo_ros_camera.so\">
          <frame_name>${name}_link</frame_name>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>
  
  <!-- Add cameras -->
  <xacro:camera_sensor name=\"head_camera\" parent_link=\"head\" position_xyz=\"0.08 0 0\" />
  
  <!-- Add IMU -->
  <link name=\"imu_link\" />
  <joint name=\"imu_joint\" type=\"fixed\">
    <parent link=\"torso\"/>
    <child link=\"imu_link\"/>
    <origin xyz=\"0 0 0.1\" rpy=\"0 0 0\"/>
  </joint>
  
  <!-- Gazebo plugin for IMU -->
  <gazebo reference=\"imu_link\">
    <sensor type=\"imu\" name=\"imu_sensor\">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type=\"gaussian\">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name=\"imu_plugin\" filename=\"libgazebo_ros_imu.so\">
        <topicName>imu/data</topicName>
        <bodyName>imu_link</bodyName>
        <updateRate>100.0</updateRate>
        <gaussianNoise>0.0</gaussianNoise>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Mathematical Expressions in XACRO

XACRO supports mathematical expressions using Python syntax:

```xml
<xacro:property name=\"pi\" value=\"3.1415926535897931\" />
<xacro:property name=\"half_pi\" value=\"${pi/2}\" />
<xacro:property name=\"sum\" value=\"${1.0 + 2.0}\" />
<xacro:property name=\"product\" value=\"${2.0 * 3.0}\" />
<xacro:property name=\"sine\" value=\"${sin(pi/4)}\" />
<xacro:property name=\"cosine\" value=\"${cos(pi/3)}\" />

<!-- Using math for calculating link properties -->
<xacro:property name=\"link_length\" value=\"0.3\" />
<xacro:property name=\"link_radius\" value=\"0.05\" />
<xacro:property name=\"link_volume\" value=\"${pi * link_radius * link_radius * link_length}\" />
<xacro:property name=\"link_mass\" value=\"${link_volume * 7000}\" />  <!-- Steel density -->
<xacro:property name=\"ixx_calc\" value=\"${link_mass * (3*link_radius*link_radius + link_length*link_length) / 12}\" />
<xacro:property name=\"izz_calc\" value=\"${link_mass * link_radius*link_radius / 2}\" />
```

## Conclusion

URDF and XACRO are fundamental tools for describing humanoid robots in ROS. URDF provides the basic structure for defining links, joints, and their physical properties, while XACRO adds powerful features like macros and variables that make it practical to create complex robot models. For humanoid robots with many similar components like arms and legs, XACRO's macro system is particularly valuable as it allows for modular, maintainable robot descriptions. Understanding these tools is crucial for anyone developing humanoid robots, as they form the foundation for simulation, visualization, and control of the robot.