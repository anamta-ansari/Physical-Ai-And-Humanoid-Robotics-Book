---
title: Projects & Assignments
sidebar_position: 1
description: ROS 2 package creation, Gazebo simulation projects, Isaac perception pipelines, VLA voice-controlled robots, and final humanoid projects
---

# Projects & Assignments

## ROS 2 package creation

Creating ROS 2 packages is fundamental to building modular and maintainable robotics applications. This section provides comprehensive projects and assignments for developing ROS 2 packages for humanoid robotics applications.

### Basic Package Structure

A standard ROS 2 package for humanoid robotics follows this structure:

```
humanoid_robot_pkg/
├── CMakeLists.txt          # Build configuration
├── package.xml            # Package metadata
├── src/                   # Source code
│   ├── controllers/
│   │   ├── walking_controller.cpp
│   │   ├── balance_controller.cpp
│   │   └── manipulation_controller.cpp
│   ├── perception/
│   │   ├── vision_processor.cpp
│   │   ├── depth_processor.cpp
│   │   └── sensor_fusion.cpp
│   └── utils/
│       ├── kinematics.cpp
│       └── trajectory_generator.cpp
├── include/               # Header files
│   └── humanoid_robot_pkg/
├── launch/                # Launch files
│   ├── robot.launch.py
│   ├── walking.launch.py
│   └── perception.launch.py
├── config/                # Configuration files
│   ├── controllers.yaml
│   └── robot_params.yaml
├── test/                  # Unit tests
│   └── test_controllers.cpp
└── scripts/               # Python scripts (if needed)
```

### Project 1: Basic ROS 2 Package for Humanoid Robot

**Objective**: Create a basic ROS 2 package that implements a simple humanoid robot interface with publishers, subscribers, and services.

#### Assignment Steps:

1. **Create the package structure**:
```bash
ros2 pkg create --build-type ament_cmake humanoid_basic_interface --dependencies rclcpp rclpy std_msgs sensor_msgs geometry_msgs builtin_interfaces
```

2. **Define the package.xml**:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_basic_interface</name>
  <version>0.1.0</version>
  <description>Basic interface for humanoid robot control and monitoring</description>
  <maintainer email="student@university.edu">Student Name</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>builtin_interfaces</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

3. **Implement the main node**:

```cpp
// include/humanoid_basic_interface/humanoid_controller.hpp
#ifndef HUMANOID_CONTROLLER_HPP
#define HUMANOID_CONTROLLER_HPP

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace humanoid_basic_interface {

class HumanoidController : public rclcpp::Node
{
public:
    HumanoidController();
    ~HumanoidController() = default;

private:
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr balance_state_publisher_;
    
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_subscriber_;
    
    // Services
    rclcpp::Service<std_msgs::msg::Bool>::SharedPtr balance_enable_service_;
    rclcpp::Service<std_msgs::msg::Bool>::SharedPtr walk_enable_service_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr control_timer_;
    
    // Robot state
    sensor_msgs::msg::JointState current_joint_state_;
    geometry_msgs::msg::Twist current_cmd_vel_;
    bool balance_enabled_;
    bool walking_enabled_;
    
    // Control parameters
    double control_frequency_;
    double max_linear_velocity_;
    double max_angular_velocity_;
    
    // Callback functions
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void jointCommandCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void balanceEnableCallback(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<std_msgs::msg::Bool::Request> request,
        std::shared_ptr<std_msgs::msg::Bool::Response> response);
    void walkEnableCallback(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<std_msgs::msg::Bool::Request> request,
        std::shared_ptr<std_msgs::msg::Bool::Response> response);
    void controlLoop();
    
    // Helper functions
    void initializeJointState();
    void publishJointState();
    void publishBalanceState();
    void updateRobotState();
    bool validateJointCommands(const sensor_msgs::msg::JointState& commands);
};

} // namespace humanoid_basic_interface

#endif // HUMANOID_CONTROLLER_HPP
```

```cpp
// src/humanoid_controller.cpp
#include "humanoid_basic_interface/humanoid_controller.hpp"
#include <chrono>
#include <cmath>

namespace humanoid_basic_interface {

HumanoidController::HumanoidController()
: Node("humanoid_controller")
{
    // Declare parameters
    this->declare_parameter("control_frequency", 100.0);
    this->declare_parameter("max_linear_velocity", 1.0);
    this->declare_parameter("max_angular_velocity", 0.5);
    
    // Get parameters
    control_frequency_ = this->get_parameter("control_frequency").as_double();
    max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
    max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
    
    // Initialize publishers
    joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "joint_states", 10);
    balance_state_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "balance_state", 10);
    
    // Initialize subscribers
    cmd_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10,
        std::bind(&HumanoidController::cmdVelCallback, this, std::placeholders::_1));
    joint_command_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "joint_commands", 10,
        std::bind(&HumanoidController::jointCommandCallback, this, std::placeholders::_1));
    
    // Initialize services
    balance_enable_service_ = this->create_service<std_msgs::msg::Bool>(
        "balance_enable",
        std::bind(&HumanoidController::balanceEnableCallback, this, 
                 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    walk_enable_service_ = this->create_service<std_msgs::msg::Bool>(
        "walk_enable",
        std::bind(&HumanoidController::walkEnableCallback, this, 
                 std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    
    // Initialize timer for control loop
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / control_frequency_)),
        std::bind(&HumanoidController::controlLoop, this));
    
    // Initialize robot state
    initializeJointState();
    
    balance_enabled_ = false;
    walking_enabled_ = false;
    
    RCLCPP_INFO(this->get_logger(), "Humanoid Controller initialized");
}

void HumanoidController::initializeJointState()
{
    // Initialize joint state message with humanoid robot joint names
    std::vector<std::string> joint_names = {
        // Left leg
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", 
        "left_knee", "left_ankle_pitch", "left_ankle_roll",
        // Right leg
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
        "right_knee", "right_ankle_pitch", "right_ankle_roll",
        // Left arm
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
        "left_elbow_pitch", "left_wrist_pitch", "left_wrist_yaw",
        // Right arm
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
        "right_elbow_pitch", "right_wrist_pitch", "right_wrist_yaw",
        // Head
        "head_yaw", "head_pitch"
    };
    
    current_joint_state_.name = joint_names;
    current_joint_state_.position.resize(joint_names.size(), 0.0);
    current_joint_state_.velocity.resize(joint_names.size(), 0.0);
    current_joint_state_.effort.resize(joint_names.size(), 0.0);
}

void HumanoidController::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    // Limit velocities
    current_cmd_vel_.linear.x = std::max(-max_linear_velocity_, 
                                        std::min(max_linear_velocity_, msg->linear.x));
    current_cmd_vel_.linear.y = std::max(-max_linear_velocity_, 
                                        std::min(max_linear_velocity_, msg->linear.y));
    current_cmd_vel_.angular.z = std::max(-max_angular_velocity_, 
                                         std::min(max_angular_velocity_, msg->angular.z));
    
    // If walking is enabled, use this velocity for gait generation
    if (walking_enabled_) {
        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,  // 1 second throttle
            "Received velocity command: linear=(%.2f, %.2f), angular=%.2f",
            current_cmd_vel_.linear.x,
            current_cmd_vel_.linear.y,
            current_cmd_vel_.angular.z
        );
    }
}

void HumanoidController::jointCommandCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (validateJointCommands(*msg)) {
        // Update joint positions based on commands
        for (size_t i = 0; i < msg->name.size(); ++i) {
            auto it = std::find(current_joint_state_.name.begin(), 
                              current_joint_state_.name.end(), msg->name[i]);
            if (it != current_joint_state_.name.end()) {
                size_t index = std::distance(current_joint_state_.name.begin(), it);
                if (index < current_joint_state_.position.size()) {
                    current_joint_state_.position[index] = msg->position[i];
                    if (msg->velocity.size() > i) {
                        current_joint_state_.velocity[index] = msg->velocity[i];
                    }
                    if (msg->effort.size() > i) {
                        current_joint_state_.effort[index] = msg->effort[i];
                    }
                }
            }
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid joint commands received");
    }
}

bool HumanoidController::validateJointCommands(const sensor_msgs::msg::JointState& commands)
{
    // Check that all joint names are valid
    for (const auto& name : commands.name) {
        if (std::find(current_joint_state_.name.begin(), 
                     current_joint_state_.name.end(), name) == 
            current_joint_state_.name.end()) {
            return false;
        }
    }
    
    // Check that position, velocity, and effort arrays match joint names
    if (commands.position.size() != commands.name.size() ||
        commands.velocity.size() > 0 && commands.velocity.size() != commands.name.size() ||
        commands.effort.size() > 0 && commands.effort.size() != commands.name.size()) {
        return false;
    }
    
    return true;
}

void HumanoidController::balanceEnableCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<std_msgs::msg::Bool::Request> request,
    std::shared_ptr<std_msgs::msg::Bool::Response> response)
{
    balance_enabled_ = request->data;
    response->data = balance_enabled_;
    
    RCLCPP_INFO(this->get_logger(), 
               "Balance control %s", 
               balance_enabled_ ? "enabled" : "disabled");
}

void HumanoidController::walkEnableCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<std_msgs::msg::Bool::Request> request,
    std::shared_ptr<std_msgs::msg::Bool::Response> response)
{
    walking_enabled_ = request->data;
    response->data = walking_enabled_;
    
    RCLCPP_INFO(this->get_logger(), 
               "Walking control %s", 
               walking_enabled_ ? "enabled" : "disabled");
}

void HumanoidController::controlLoop()
{
    // Update robot state based on current commands
    updateRobotState();
    
    // Publish current joint states
    publishJointState();
    
    // Publish balance state if enabled
    if (balance_enabled_) {
        publishBalanceState();
    }
}

void HumanoidController::updateRobotState()
{
    // In a real implementation, this would update the robot's state
    // based on control algorithms, sensor feedback, etc.
    // For this example, we'll just simulate state changes
    
    if (walking_enabled_ && std::abs(current_cmd_vel_.linear.x) > 0.01) {
        // Simulate walking motion - add oscillation to leg joints
        static double walk_phase = 0.0;
        walk_phase += 0.1;  // Increment phase
        
        // Apply walking gait pattern to leg joints
        for (size_t i = 0; i < current_joint_state_.name.size(); ++i) {
            const std::string& joint_name = current_joint_state_.name[i];
            
            if (joint_name.find("hip_pitch") != std::string::npos ||
                joint_name.find("knee") != std::string::npos ||
                joint_name.find("ankle_pitch") != std::string::npos) {
                
                // Apply walking pattern
                double amplitude = (joint_name.find("left") != std::string::npos) ? 1.0 : -1.0;
                current_joint_state_.position[i] += 0.1 * sin(walk_phase) * amplitude;
            }
        }
    }
}

void HumanoidController::publishJointState()
{
    current_joint_state_.header.stamp = this->get_clock()->now();
    current_joint_state_.header.frame_id = "base_link";
    
    joint_state_publisher_->publish(current_joint_state_);
}

void HumanoidController::publishBalanceState()
{
    std_msgs::msg::Float64MultiArray balance_msg;
    
    // Calculate balance metrics (simplified example)
    std::vector<double> balance_data = {
        0.0,  // Center of mass x
        0.0,  // Center of mass y
        0.8,  // Center of mass z (height)
        0.0,  // Zero moment point x
        0.0,  // Zero moment point y
        0.0,  // Angular momentum x
        0.0,  // Angular momentum y
        0.0   // Angular momentum z
    };
    
    balance_msg.data = balance_data;
    balance_state_publisher_->publish(balance_msg);
}

} // namespace humanoid_basic_interface

// Main function
#include "rclcpp/rclcpp.hpp"
#include "humanoid_basic_interface/humanoid_controller.hpp"

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<humanoid_basic_interface::HumanoidController>());
    rclcpp::shutdown();
    return 0;
}
```

4. **Update CMakeLists.txt**:

```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_basic_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Include directories
include_directories(include)

# Create executable
add_executable(humanoid_controller
  src/humanoid_controller.cpp
  src/main.cpp
)

# Link libraries
ament_target_dependencies(humanoid_controller
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  builtin_interfaces
)

# Install targets
install(TARGETS
  humanoid_controller
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

5. **Create a launch file**:

```python
# launch/humanoid_basic_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        
        DeclareLaunchArgument(
            'control_frequency',
            default_value='100.0',
            description='Control frequency in Hz'
        ),
        
        # Humanoid controller node
        Node(
            package='humanoid_basic_interface',
            executable='humanoid_controller',
            name='humanoid_controller',
            parameters=[
                {'control_frequency': LaunchConfiguration('control_frequency')},
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ],
            output='screen'
        )
    ])
```

### Project 2: Advanced ROS 2 Package with Actions and Services

**Objective**: Create an advanced ROS 2 package that implements actions for complex humanoid behaviors and services for robot management.

```python
# Example action server implementation
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from humanoid_msgs.action import WalkToPose, ManipulateObject
import math
import time

class HumanoidAdvancedController(Node):
    def __init__(self):
        super().__init__('humanoid_advanced_controller')
        
        # Create action servers
        self.walk_to_pose_server = ActionServer(
            self,
            WalkToPose,
            'walk_to_pose',
            self.execute_walk_to_pose
        )
        
        self.manipulate_object_server = ActionServer(
            self,
            ManipulateObject,
            'manipulate_object',
            self.execute_manipulate_object
        )
        
        # Create service servers
        self.reset_service = self.create_service(
            ResetRobot, 'reset_robot', self.reset_robot_callback
        )
        
        # Robot state
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.is_moving = False
        self.is_manipulating = False
        
        self.get_logger().info('Humanoid Advanced Controller initialized')
    
    def execute_walk_to_pose(self, goal_handle):
        """
        Execute walk to pose action
        """
        self.get_logger().info('Executing walk to pose action...')
        
        target_pose = goal_handle.request.target_pose
        tolerance = goal_handle.request.tolerance
        
        # Initialize feedback
        feedback_msg = WalkToPose.Feedback()
        result = WalkToPose.Result()
        
        # Simulate walking to target
        current_pos = self.current_pose.copy()
        step_size = 0.05  # 5cm steps
        max_steps = 1000  # Safety limit
        
        for step in range(max_steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal was canceled'
                return result
            
            # Calculate direction to target
            dx = target_pose.position.x - current_pos[0]
            dy = target_pose.position.y - current_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < tolerance:
                # Reached target
                self.current_pose[0] = target_pose.position.x
                self.current_pose[1] = target_pose.position.y
                self.current_pose[2] = math.atan2(dy, dx)  # Face target direction
                
                goal_handle.succeed()
                result.success = True
                result.message = f'Reached target in {step} steps'
                return result
            
            # Move one step toward target
            if distance > 0:
                step_x = current_pos[0] + (dx / distance) * step_size
                step_y = current_pos[1] + (dy / distance) * step_size
                current_pos[0] = step_x
                current_pos[1] = step_y
                current_pos[2] = math.atan2(dy, dx)
            
            # Publish feedback
            feedback_msg.current_pose.position.x = current_pos[0]
            feedback_msg.current_pose.position.y = current_pos[1]
            feedback_msg.distance_remaining = distance
            goal_handle.publish_feedback(feedback_msg)
            
            # Sleep to simulate real movement
            time.sleep(0.1)
        
        # Failed to reach target within limits
        goal_handle.abort()
        result.success = False
        result.message = f'Failed to reach target after {max_steps} steps'
        return result
    
    def execute_manipulate_object(self, goal_handle):
        """
        Execute manipulate object action
        """
        self.get_logger().info(f'Executing manipulation of object: {goal_handle.request.object_id}')
        
        # Initialize feedback
        feedback_msg = ManipulateObject.Feedback()
        result = ManipulateObject.Result()
        
        # Simulate manipulation steps
        manipulation_steps = [
            'approaching_object',
            'grasping_object', 
            'lifting_object',
            'moving_object',
            'placing_object',
            'retracting_arm'
        ]
        
        for i, step in enumerate(manipulation_steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = f'Manipulation canceled during {step}'
                return result
            
            # Update feedback
            feedback_msg.current_step = step
            feedback_msg.progress_percentage = (i + 1) / len(manipulation_steps) * 100.0
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate step execution
            time.sleep(1.0)
        
        # Complete manipulation
        goal_handle.succeed()
        result.success = True
        result.message = f'Successfully manipulated object {goal_handle.request.object_id}'
        return result
    
    def reset_robot_callback(self, request, response):
        """
        Reset robot to safe state
        """
        self.get_logger().info('Resetting robot to safe state')
        
        # Stop all motion
        self.is_moving = False
        self.is_manipulating = False
        
        # Return to home position
        self.current_pose = [0.0, 0.0, 0.0]
        
        # Reset any other state as needed
        response.success = True
        response.message = 'Robot reset to safe state'
        
        return response

def main(args=None):
    rclpy.init(args=args)
    
    controller = HumanoidAdvancedController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down humanoid advanced controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Gazebo simulation project

This project focuses on creating and simulating humanoid robots in Gazebo, a popular robotics simulator.

### Project 3: Humanoid Robot Model in Gazebo

**Objective**: Create a complete humanoid robot model with URDF, SDF, and Gazebo plugins, then simulate walking and basic manipulation.

#### Assignment Steps:

1. **Create the URDF model**:

```xml
<!-- urdf/humanoid_robot.urdf -->
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  
  <!-- Materials -->
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="blue">
    <color rgba="0.2 0.2 1 1"/>
  </material>
  <material name="green">
    <color rgba="0.1 0.8 0.1 1"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
  <material name="orange">
    <color rgba="1 0.423529411765 0.0392156862745 1"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.6" radius="0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.6" radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <link name="head_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left leg -->
  <joint name="left_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip_yaw_link"/>
    <origin xyz="0.05 0.1 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3"/>
  </joint>

  <link name="left_hip_yaw_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_hip_roll_joint" type="revolute">
    <parent link="left_hip_yaw_link"/>
    <child link="left_hip_roll_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="3"/>
  </joint>

  <link name="left_hip_roll_link">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Continue for all joints and links... -->
  <!-- (Additional joints and links for full humanoid model would be defined here) -->

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Left leg visual and collision properties -->
  <gazebo reference="left_hip_yaw_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <gazebo reference="left_hip_roll_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <!-- Add similar gazebo properties for all links -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

  <gazebo reference="head_link">
    <material>Gazebo/White</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
  </gazebo>

</robot>
```

2. **Create controller configuration**:

```yaml
# config/controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz
    use_sim_time: true

    humanoid_joint_publisher:
      type: joint_state_controller/JointStateController

    left_leg_controller:
      type: position_controllers/JointTrajectoryController

    right_leg_controller:
      type: position_controllers/JointTrajectoryController

    left_arm_controller:
      type: position_controllers/JointTrajectoryController

    right_arm_controller:
      type: position_controllers/JointTrajectoryController

humanoid_joint_publisher:
  ros__parameters:
    use_sim_time: true

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_yaw_joint
      - left_hip_roll_joint
      - left_hip_pitch_joint
      - left_knee_joint
      - left_ankle_pitch_joint
      - left_ankle_roll_joint
    interface_name: position
    use_sim_time: true

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_yaw_joint
      - right_hip_roll_joint
      - right_hip_pitch_joint
      - right_knee_joint
      - right_ankle_pitch_joint
      - right_ankle_roll_joint
    interface_name: position
    use_sim_time: true

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch_joint
      - left_shoulder_roll_joint
      - left_shoulder_yaw_joint
      - left_elbow_joint
      - left_wrist_pitch_joint
      - left_wrist_yaw_joint
    interface_name: position
    use_sim_time: true

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_pitch_joint
      - right_shoulder_roll_joint
      - right_shoulder_yaw_joint
      - right_elbow_joint
      - right_wrist_pitch_joint
      - right_wrist_yaw_joint
    interface_name: position
    use_sim_time: true
```

3. **Create launch file for simulation**:

```python
# launch/humanoid_gazebo.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),
        
        # Include Gazebo launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'verbose': 'false',
                'pause': 'false',
            }.items()
        ),
        
        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', robot_name,
                '-x', '0', '-y', '0', '-z', '0.85'  # Start above ground for humanoid
            ],
            output='screen'
        ),
        
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': PathJoinSubstitution([
                    FindPackageShare('humanoid_gazebo'),
                    'urdf',
                    'humanoid_robot.urdf'
                ])}
            ]
        ),
        
        # Joint state publisher (for GUI control)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        
        # Controller manager
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('humanoid_gazebo'),
                    'config',
                    'controllers.yaml'
                ]),
                {'use_sim_time': use_sim_time}
            ],
            output='both'
        ),
        
        # Load controllers
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['humanoid_joint_publisher'],
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['left_leg_controller'],
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['right_leg_controller'],
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['left_arm_controller'],
            parameters=[{'use_sim_time': use_sim_time}]
        ),
        
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['right_arm_controller'],
            parameters=[{'use_sim_time': use_sim_time}]
        )
    ])
```

4. **Create walking controller**:

```python
# scripts/walking_controller.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
import math
import time

class WalkingController(Node):
    def __init__(self):
        super().__init__('walking_controller')
        
        # Publisher for joint trajectory commands
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 
            '/left_leg_controller/joint_trajectory', 
            10
        )
        
        # Publisher for walking commands
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Walking parameters
        self.declare_parameter('step_length', 0.3)
        self.declare_parameter('step_height', 0.05)
        self.declare_parameter('step_duration', 0.8)
        self.declare_parameter('walking_frequency', 0.5)
        
        self.step_length = self.get_parameter('step_length').value
        self.step_height = self.get_parameter('step_height').value
        self.step_duration = self.get_parameter('step_duration').value
        self.walking_frequency = self.get_parameter('walking_frequency').value
        
        # Walking state
        self.is_walking = False
        self.walk_direction = 1.0  # 1 for forward, -1 for backward
        self.current_phase = 0.0
        
        # Timer for walking control
        self.walk_timer = self.create_timer(0.01, self.walk_control_callback)
        
        self.get_logger().info('Walking controller initialized')
    
    def cmd_vel_callback(self, msg):
        """
        Handle velocity commands for walking
        """
        linear_x = msg.linear.x
        
        if abs(linear_x) > 0.01:  # Threshold for walking
            self.is_walking = True
            self.walk_direction = 1.0 if linear_x > 0 else -1.0
        else:
            self.is_walking = False
    
    def walk_control_callback(self):
        """
        Main walking control callback
        """
        if not self.is_walking:
            return
        
        # Update walking phase
        self.current_phase += 2 * math.pi * self.walking_frequency * 0.01  # 0.01s timer
        if self.current_phase > 2 * math.pi:
            self.current_phase -= 2 * math.pi
        
        # Generate walking pattern
        trajectory_msg = self.generate_walking_trajectory(self.current_phase)
        
        if trajectory_msg:
            self.trajectory_publisher.publish(trajectory_msg)
    
    def generate_walking_trajectory(self, phase):
        """
        Generate walking trajectory based on phase
        """
        # Define joint names for the leg
        joint_names = [
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint'
        ]
        
        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = joint_names
        trajectory_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        
        # Generate walking gait pattern
        # This is a simplified example - real walking would be more complex
        base_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Neutral positions
        
        # Apply walking pattern to positions
        # Hip pitch oscillates with walking phase
        hip_pitch_offset = 0.1 * math.sin(phase) * self.walk_direction
        knee_offset = 0.15 * math.sin(phase + math.pi/2) * self.walk_direction  # Phase shifted
        
        positions = base_positions.copy()
        positions[2] = base_positions[2] + hip_pitch_offset  # Hip pitch
        positions[3] = base_positions[3] + knee_offset      # Knee
        
        point.positions = positions
        point.velocities = [0.0] * len(positions)  # Zero velocities for simplicity
        point.accelerations = [0.0] * len(positions)  # Zero accelerations for simplicity
        
        # Set time from start (in the future to execute after a delay)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000  # 50ms in the future
        
        trajectory_msg.points = [point]
        
        return trajectory_msg

def main(args=None):
    rclpy.init(args=args)
    
    controller = WalkingController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down walking controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac perception pipeline

NVIDIA Isaac provides a powerful platform for robotics perception, particularly for humanoid robots that need to understand their environment.

### Project 4: Isaac Perception Pipeline

**Objective**: Implement a perception pipeline using Isaac Sim and Isaac ROS for humanoid robot perception tasks.

```python
# isaac_perception_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class IsaacPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('isaac_perception_pipeline')
        
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
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10
        )
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/isaac_detections', 10
        )
        
        self.feature_pub = self.create_publisher(
            PointStamped, '/isaac_features', 10
        )
        
        # Initialize perception models
        self.initialize_perception_models()
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        self.get_logger().info('Isaac Perception Pipeline initialized')
    
    def initialize_perception_models(self):
        """
        Initialize Isaac-compatible perception models
        """
        try:
            # Initialize object detection model
            # Using a pre-trained model for demonstration
            self.detection_model = torch.hub.load(
                'ultralytics/yolov5', 'yolov5s', pretrained=True
            ).to(self.device)
            self.detection_model.eval()
            
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
            
            self.get_logger().info('Perception models loaded successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load perception models: {str(e)}')
            raise
    
    def image_callback(self, msg):
        """
        Process incoming image with Isaac perception pipeline
        """
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process with Isaac perception pipeline
            start_time = self.get_clock().now().nanoseconds / 1e9
            
            # Run object detection
            detections = self.run_object_detection(cv_image)
            
            # Extract features
            features = self.extract_features(cv_image)
            
            # Estimate depth if camera parameters available
            depth_map = None
            if self.camera_matrix is not None:
                depth_map = self.estimate_depth(cv_image)
            
            # Calculate processing time
            end_time = self.get_clock().now().nanoseconds / 1e9
            processing_time = (end_time - start_time) * 1000  # milliseconds
            
            # Publish results
            if detections:
                detections.header = msg.header
                self.detection_publisher.publish(detections)
            
            if features:
                features.header = msg.header
                self.feature_publisher.publish(features)
            
            # Track performance
            self.frame_count += 1
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.frame_count % 100 == 0:
                avg_fps = self.frame_count / (current_time - self.start_time)
                self.get_logger().info(
                    f'Processed {self.frame_count} frames. '
                    f'Avg FPS: {avg_fps:.2f}, Processing time: {processing_time:.2f}ms'
                )
                
        except Exception as e:
            self.get_logger().error(f'Error in perception pipeline: {str(e)}')
    
    def run_object_detection(self, image):
        """
        Run object detection using Isaac-compatible model
        """
        try:
            # Preprocess image for model
            input_tensor = self.preprocess_image_for_detection(image)
            
            # Run inference
            with torch.no_grad():
                results = self.detection_model(input_tensor)
            
            # Process results
            detections = self.process_detection_results(results, image.shape)
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')
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
    
    def process_detection_results(self, results, image_shape):
        """
        Process raw detection results into ROS messages
        """
        # Get detections (results.pred[0] contains the detections for batch 0)
        detections = results.pred[0]
        
        if detections is None or len(detections) == 0:
            # Return empty detection array
            empty_detections = Detection2DArray()
            empty_detections.header.frame_id = 'camera_rgb_optical_frame'
            return empty_detections
        
        # Convert to Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header.frame_id = 'camera_rgb_optical_frame'
        
        height, width = image_shape[:2]
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
    
    def extract_features(self, image):
        """
        Extract features using Isaac-compatible feature model
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
                
                # Convert to numpy for ROS message
                feature_np = feature_vector.cpu().numpy()[0]  # Take first batch item
                
                # Create PointStamped message to hold features
                features_msg = PointStamped()
                features_msg.point.x = float(feature_np[0]) if len(feature_np) > 0 else 0.0
                features_msg.point.y = float(feature_np[1]) if len(feature_np) > 1 else 0.0
                features_msg.point.z = float(feature_np[2]) if len(feature_np) > 2 else 0.0
                
                return features_msg
        
        except Exception as e:
            self.get_logger().error(f'Error in feature extraction: {str(e)}')
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
    
    def estimate_depth(self, image):
        """
        Estimate depth using Isaac-compatible depth model
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
            self.get_logger().error(f'Error in depth estimation: {str(e)}')
            return None
    
    def preprocess_image_for_depth(self, image):
        """
        Preprocess image for depth estimation model
        """
        # Resize to model input size (256x256 for MiDaS small)
        input_size = (256, 256)
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
    
    def camera_info_callback(self, msg):
        """
        Store camera information for depth processing
        """
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacPerceptionPipeline()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac Perception Pipeline')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## VLA voice-controlled robot

Vision-Language-Action (VLA) systems enable robots to understand and respond to natural language commands in visual contexts.

### Project 5: VLA Voice-Controlled Robot

**Objective**: Create a robot that can receive voice commands, understand visual context, and execute appropriate actions.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from audio_common_msgs.msg import AudioData
import speech_recognition as sr
import openai
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
import queue

class VLAVoiceControlledRobot(Node):
    def __init__(self):
        super().__init__('vla_voice_controlled_robot')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # OpenAI API setup (for natural language understanding)
        # Note: In practice, you'd use a more efficient local model
        # For this example, we'll simulate the understanding process
        self.use_local_nlu = True  # Use local model instead of cloud API
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        
        self.audio_sub = self.create_subscription(
            AudioData, '/audio/raw', self.audio_callback, 10
        )
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        
        # Robot state
        self.current_image = None
        self.command_queue = queue.Queue()
        self.is_processing = False
        
        # Voice command processing
        self.voice_command_thread = threading.Thread(
            target=self.process_voice_commands, daemon=True
        )
        self.voice_command_thread.start()
        
        # Command mapping
        self.command_mapping = {
            'move forward': 'move_forward',
            'go forward': 'move_forward',
            'move backward': 'move_backward',
            'go backward': 'move_backward',
            'turn left': 'turn_left',
            'turn right': 'turn_right',
            'stop': 'stop',
            'go to the kitchen': 'navigate_to_kitchen',
            'find the red cup': 'find_red_cup',
            'pick up the object': 'grasp_object',
            'put down the object': 'place_object',
            'dance': 'dance',
            'introduce yourself': 'introduce'
        }
        
        self.get_logger().info('VLA Voice-Controlled Robot initialized')
    
    def image_callback(self, msg):
        """
        Store current image for visual context
        """
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
    
    def audio_callback(self, msg):
        """
        Process audio data for voice commands
        """
        # Convert audio data to audio file format for processing
        try:
            # In practice, you'd convert the AudioData message to a format
            # that speech recognition can process
            # For this example, we'll simulate the recognition process
            
            # Extract audio from message (simplified)
            audio_data = np.frombuffer(msg.data, dtype=np.int16)
            
            # Process with speech recognition
            command = self.recognize_speech(audio_data)
            
            if command:
                self.get_logger().info(f'Recognized command: {command}')
                
                # Add command to processing queue
                self.command_queue.put(command)
                
        except Exception as e:
            self.get_logger().error(f'Error processing audio: {str(e)}')
    
    def recognize_speech(self, audio_data):
        """
        Recognize speech from audio data
        """
        try:
            # In a real implementation, you'd use the actual audio data
            # For this example, we'll return a simulated recognition result
            # based on the audio data characteristics
            
            # Convert numpy array back to audio format for recognizer
            # This is a simplified simulation
            if len(audio_data) > 1000:  # If we have sufficient audio data
                # In practice, you'd use speech recognition libraries
                # like SpeechRecognition with PyAudio or similar
                return self.simulate_speech_recognition(audio_data)
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error in speech recognition: {str(e)}')
            return None
    
    def simulate_speech_recognition(self, audio_data):
        """
        Simulate speech recognition (in practice, use real SR)
        """
        # This is a simulation - in practice, use actual speech recognition
        # For example: Google Speech Recognition, Whisper, or similar
        return "move forward"  # Simulated result
    
    def process_voice_commands(self):
        """
        Process voice commands in a separate thread
        """
        while rclpy.ok():
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=1.0)
                
                if command:
                    # Process the command
                    self.execute_command(command)
                
            except queue.Empty:
                continue  # No command in queue, continue loop
            except Exception as e:
                self.get_logger().error(f'Error processing voice command: {str(e)}')
    
    def execute_command(self, command_text):
        """
        Execute a command based on recognized text
        """
        if self.is_processing:
            self.get_logger().warn('Already processing a command, skipping')
            return
        
        self.is_processing = True
        
        try:
            # Map command to action
            action = self.map_command_to_action(command_text.lower())
            
            if action:
                self.get_logger().info(f'Executing action: {action}')
                
                # Execute the mapped action
                self.perform_action(action)
                
                # Provide feedback
                self.provide_feedback(action)
            else:
                self.get_logger().warn(f'Unknown command: {command_text}')
                self.speak_response("I don't understand that command. Please try again.")
        
        except Exception as e:
            self.get_logger().error(f'Error executing command {command_text}: {str(e)}')
            self.speak_response("I encountered an error processing your command.")
        finally:
            self.is_processing = False
    
    def map_command_to_action(self, command_text):
        """
        Map natural language command to robot action
        """
        # Check for direct matches
        if command_text in self.command_mapping:
            return self.command_mapping[command_text]
        
        # Check for partial matches
        for cmd, action in self.command_mapping.items():
            if cmd in command_text:
                return action
        
        # For more complex understanding, you'd use NLU models
        # Here we'll implement some basic pattern matching
        if 'move' in command_text or 'go' in command_text:
            if 'forward' in command_text:
                return 'move_forward'
            elif 'backward' in command_text or 'back' in command_text:
                return 'move_backward'
            elif 'left' in command_text:
                return 'turn_left'
            elif 'right' in command_text:
                return 'turn_right'
        
        if 'stop' in command_text:
            return 'stop'
        
        if 'find' in command_text or 'look' in command_text:
            if 'red' in command_text:
                return 'find_red_object'
            elif 'blue' in command_text:
                return 'find_blue_object'
            else:
                return 'find_object'
        
        if 'pick up' in command_text or 'grasp' in command_text or 'take' in command_text:
            return 'grasp_object'
        
        if 'put' in command_text or 'place' in command_text or 'drop' in command_text:
            return 'place_object'
        
        return None  # Unknown command
    
    def perform_action(self, action):
        """
        Perform the specified action
        """
        if action == 'move_forward':
            self.move_forward()
        elif action == 'move_backward':
            self.move_backward()
        elif action == 'turn_left':
            self.turn_left()
        elif action == 'turn_right':
            self.turn_right()
        elif action == 'stop':
            self.stop_movement()
        elif action == 'navigate_to_kitchen':
            self.navigate_to_location('kitchen')
        elif action == 'find_red_cup':
            self.find_object('red cup')
        elif action == 'grasp_object':
            self.grasp_object()
        elif action == 'place_object':
            self.place_object()
        elif action == 'dance':
            self.perform_dance()
        elif action == 'introduce':
            self.introduce_robot()
        elif action == 'find_red_object':
            self.find_object('red')
        elif action == 'find_blue_object':
            self.find_object('blue')
        elif action == 'find_object':
            self.find_object('any')
    
    def move_forward(self):
        """
        Move robot forward
        """
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward at 0.3 m/s
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Moving forward')
    
    def move_backward(self):
        """
        Move robot backward
        """
        cmd = Twist()
        cmd.linear.x = -0.3  # Move backward at 0.3 m/s
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Moving backward')
    
    def turn_left(self):
        """
        Turn robot left
        """
        cmd = Twist()
        cmd.angular.z = 0.5  # Turn left at 0.5 rad/s
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Turning left')
    
    def turn_right(self):
        """
        Turn robot right
        """
        cmd = Twist()
        cmd.angular.z = -0.5  # Turn right at 0.5 rad/s
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Turning right')
    
    def stop_movement(self):
        """
        Stop robot movement
        """
        cmd = Twist()
        # Zero velocities (default values)
        self.cmd_vel_publisher.publish(cmd)
        self.get_logger().info('Stopping movement')
    
    def navigate_to_location(self, location):
        """
        Navigate to a specific location
        """
        # This would integrate with navigation stack
        # For this example, we'll simulate navigation
        self.get_logger().info(f'Navigating to {location}')
        
        # In practice, you'd use navigation2 stack:
        # - Send goal to Nav2
        # - Monitor progress
        # - Handle navigation failures
        pass
    
    def find_object(self, object_description):
        """
        Find an object based on description
        """
        if self.current_image is None:
            self.get_logger().warn('No image available for object detection')
            self.speak_response("I need to see to find objects.")
            return
        
        # Process current image to find objects
        found_objects = self.detect_objects_in_image(self.current_image, object_description)
        
        if found_objects:
            self.get_logger().info(f'Found {len(found_objects)} instances of {object_description}')
            self.speak_response(f"I found the {object_description}.")
            
            # Highlight found objects in image (for visualization)
            self.highlight_found_objects(found_objects)
        else:
            self.get_logger().info(f'Could not find {object_description}')
            self.speak_response(f"I couldn't find the {object_description}.")
    
    def detect_objects_in_image(self, image, object_description):
        """
        Detect objects in image based on description
        """
        # This would use object detection models
        # For this example, we'll simulate detection
        
        # In practice, you'd use models like:
        # - YOLO for general object detection
        # - Segment Anything for flexible object segmentation
        # - CLIP for text-conditioned detection
        
        # Simulate detection based on description
        if 'red' in object_description:
            # Simulate finding red objects
            return [{'bbox': [100, 100, 200, 200], 'confidence': 0.85, 'class': 'red_object'}]
        elif 'blue' in object_description:
            # Simulate finding blue objects
            return [{'bbox': [300, 200, 400, 300], 'confidence': 0.78, 'class': 'blue_object'}]
        else:
            # Simulate finding any object
            return [{'bbox': [200, 150, 300, 250], 'confidence': 0.92, 'class': 'generic_object'}]
    
    def highlight_found_objects(self, objects):
        """
        Highlight found objects in the current image
        """
        if self.current_image is not None:
            image_copy = self.current_image.copy()
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_copy, f"{obj['class']}: {obj['confidence']:.2f}",
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # In practice, you'd publish this annotated image
            # for visualization purposes
    
    def grasp_object(self):
        """
        Grasp an object
        """
        self.get_logger().info('Attempting to grasp object')
        self.speak_response("Attempting to grasp object.")
        
        # This would involve:
        # 1. Identifying closest graspable object
        # 2. Planning grasp trajectory
        # 3. Executing grasp with manipulator
        # 4. Verifying grasp success
        pass
    
    def place_object(self):
        """
        Place held object
        """
        self.get_logger().info('Attempting to place object')
        self.speak_response("Attempting to place object.")
        
        # This would involve:
        # 1. Identifying suitable placement location
        # 2. Planning placement trajectory
        # 3. Executing placement with manipulator
        # 4. Releasing object
        pass
    
    def perform_dance(self):
        """
        Perform a simple dance routine
        """
        self.get_logger().info('Performing dance routine')
        self.speak_response("Dancing for you!")
        
        # This would involve:
        # 1. Playing music (if available)
        # 2. Executing choreographed movements
        # 3. Coordinating with rhythm
        pass
    
    def introduce_robot(self):
        """
        Introduce the robot
        """
        intro_text = "Hello! I am a humanoid robot designed to assist with various tasks. I can understand voice commands, perceive my environment, and perform actions like navigation and manipulation."
        self.speak_response(intro_text)
        self.get_logger().info('Introducing robot')
    
    def speak_response(self, text):
        """
        Publish speech response
        """
        response_msg = String()
        response_msg.data = text
        self.speech_publisher.publish(response_msg)

def main(args=None):
    rclpy.init(args=args)
    
    node = VLAVoiceControlledRobot()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLA Voice-Controlled Robot')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Final humanoid project

The capstone project integrates all the concepts learned throughout the course into a comprehensive humanoid robot application.

### Project 6: Autonomous Humanoid Robot

**Objective**: Develop a complete humanoid robot system that can receive voice commands, plan actions, navigate environments, identify objects, manipulate them, and complete tasks autonomously.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, JointState, Imu
from nav_msgs.msg import Odometry
from humanoid_msgs.msg import BalanceState
from humanoid_msgs.srv import ExecuteAction
import numpy as np
import time
import threading
import queue
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    LISTENING = 2
    PROCESSING = 3
    NAVIGATING = 4
    MANIPULATING = 5
    BALANCING = 6
    EMERGENCY_STOP = 7

class AutonomousHumanoidRobot(Node):
    def __init__(self):
        super().__init__('autonomous_humanoid_robot')
        
        # Initialize robot state
        self.robot_state = RobotState.IDLE
        self.current_pose = np.zeros(3)  # x, y, theta
        self.current_balance = np.zeros(2)  # x, y CoM offset
        self.joint_positions = {}
        self.command_queue = queue.Queue()
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/robot_speech', 10)
        self.action_pub = self.create_publisher(String, '/robot_action', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.balance_sub = self.create_subscription(BalanceState, '/balance_state', self.balance_callback, 10)
        self.voice_command_sub = self.create_subscription(String, '/voice_command', self.voice_command_callback, 10)
        
        # Services
        self.execute_action_srv = self.create_service(ExecuteAction, 'execute_action', self.execute_action_callback)
        
        # Robot capabilities
        self.capabilities = {
            'navigation': True,
            'manipulation': False,  # Assuming basic humanoid without manipulation for this example
            'voice_interaction': True,
            'balance_control': True
        }
        
        # Task execution thread
        self.task_executor_thread = threading.Thread(target=self.execute_queued_tasks, daemon=True)
        self.task_executor_thread.start()
        
        # Emergency stop handling
        self.emergency_stop_active = False
        self.emergency_stop_sub = self.create_subscription(String, '/emergency_stop', self.emergency_stop_callback, 10)
        
        # State monitoring timer
        self.state_monitor_timer = self.create_timer(0.1, self.monitor_robot_state)
        
        self.get_logger().info('Autonomous Humanoid Robot initialized')
    
    def odom_callback(self, msg):
        """
        Update robot's current pose from odometry
        """
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        # Extract theta from quaternion (simplified - assumes only rotation around Z)
        from math import atan2
        q = msg.pose.pose.orientation
        self.current_pose[2] = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    def joint_state_callback(self, msg):
        """
        Update joint positions
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
    
    def imu_callback(self, msg):
        """
        Update IMU-based state
        """
        # Use IMU data for balance monitoring
        pass
    
    def balance_callback(self, msg):
        """
        Update balance state
        """
        self.current_balance[0] = msg.com_x_offset
        self.current_balance[1] = msg.com_y_offset
    
    def voice_command_callback(self, msg):
        """
        Handle voice commands
        """
        command = msg.data.lower().strip()
        
        if command:
            self.get_logger().info(f'Received voice command: {command}')
            
            # Parse command and add to execution queue
            parsed_task = self.parse_voice_command(command)
            if parsed_task:
                self.command_queue.put(parsed_task)
    
    def parse_voice_command(self, command):
        """
        Parse natural language command into executable task
        """
        # Define command patterns
        command_patterns = {
            'navigation': [
                ('go to (.+)', 'navigate_to'),
                ('move to (.+)', 'navigate_to'),
                ('walk to (.+)', 'navigate_to'),
                ('move forward', 'move_forward'),
                ('go forward', 'move_forward'),
                ('move backward', 'move_backward'),
                ('go backward', 'move_backward'),
                ('turn left', 'turn_left'),
                ('turn right', 'turn_right'),
                ('rotate left', 'turn_left'),
                ('rotate right', 'turn_right')
            ],
            'manipulation': [
                ('pick up (.+)', 'pick_up'),
                ('grasp (.+)', 'grasp'),
                ('take (.+)', 'take'),
                ('place (.+)', 'place'),
                ('put (.+)', 'place'),
                ('drop (.+)', 'drop')
            ],
            'interaction': [
                ('say (.+)', 'speak'),
                ('hello', 'greet'),
                ('introduce yourself', 'introduce'),
                ('what can you do', 'capabilities'),
                ('stop', 'stop')
            ]
        }
        
        # Try to match command to patterns
        for category, patterns in command_patterns.items():
            for pattern, action in patterns:
                import re
                match = re.search(pattern, command)
                if match:
                    if category == 'navigation':
                        if action == 'navigate_to':
                            return {'type': 'navigation', 'action': action, 'target': match.group(1)}
                        else:
                            return {'type': 'navigation', 'action': action}
                    elif category == 'manipulation':
                        return {'type': 'manipulation', 'action': action, 'object': match.group(1)}
                    elif category == 'interaction':
                        if action == 'speak':
                            return {'type': 'interaction', 'action': action, 'text': match.group(1)}
                        else:
                            return {'type': 'interaction', 'action': action}
        
        # If no pattern matches, return as unrecognized
        return {'type': 'unrecognized', 'command': command}
    
    def execute_action_callback(self, request, response):
        """
        Service callback for executing actions
        """
        try:
            # Execute the requested action
            success = self.execute_action_directly(request.action_type, request.action_params)
            
            response.success = success
            response.message = "Action completed successfully" if success else "Action failed"
            
        except Exception as e:
            response.success = False
            response.message = f"Error executing action: {str(e)}"
        
        return response
    
    def execute_action_directly(self, action_type, params):
        """
        Execute an action directly (not from voice command)
        """
        if action_type == 'walk_forward':
            return self.execute_walk_forward(params.get('distance', 1.0))
        elif action_type == 'turn':
            return self.execute_turn(params.get('angle', 90.0))
        elif action_type == 'balance':
            return self.execute_balance_correction()
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            return False
    
    def execute_walk_forward(self, distance):
        """
        Execute walking forward for a specified distance
        """
        # Calculate required movement time based on current velocity
        target_velocity = 0.3  # m/s
        movement_time = distance / target_velocity
        
        cmd = Twist()
        cmd.linear.x = target_velocity
        
        # Move for calculated time
        start_time = time.time()
        while time.time() - start_time < movement_time and not self.emergency_stop_active:
            self.cmd_vel_publisher.publish(cmd)
            time.sleep(0.01)  # 10ms sleep
        
        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        
        return True
    
    def execute_turn(self, angle_degrees):
        """
        Execute turning for a specified angle
        """
        angle_rad = np.deg2rad(angle_degrees)
        target_angular_velocity = 0.5  # rad/s
        turn_time = abs(angle_rad) / target_angular_velocity
        
        cmd = Twist()
        cmd.angular.z = target_angular_velocity if angle_rad > 0 else -target_angular_velocity
        
        # Turn for calculated time
        start_time = time.time()
        while time.time() - start_time < turn_time and not self.emergency_stop_active:
            self.cmd_vel_publisher.publish(cmd)
            time.sleep(0.01)  # 10ms sleep
        
        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)
        
        return True
    
    def execute_balance_correction(self):
        """
        Execute balance correction maneuver
        """
        if np.linalg.norm(self.current_balance) > 0.05:  # If CoM offset > 5cm
            # Calculate required hip adjustments to correct balance
            hip_cmd = Twist()
            hip_cmd.linear.y = -self.current_balance[1] * 10  # Correct lateral imbalance
            hip_cmd.angular.y = -self.current_balance[0] * 5   # Correct forward/back imbalance
            
            # Apply correction for a short duration
            start_time = time.time()
            while time.time() - start_time < 1.0 and not self.emergency_stop_active:
                self.cmd_vel_publisher.publish(hip_cmd)
                time.sleep(0.01)
            
            # Return to neutral
            neutral_cmd = Twist()
            self.cmd_vel_publisher.publish(neutral_cmd)
        
        return True
    
    def execute_queued_tasks(self):
        """
        Execute tasks from the command queue in a separate thread
        """
        while rclpy.ok():
            try:
                # Get task from queue
                task = self.command_queue.get(timeout=1.0)
                
                if task:
                    self.execute_single_task(task)
                
            except queue.Empty:
                continue  # No task in queue, continue loop
            except Exception as e:
                self.get_logger().error(f'Error executing queued task: {str(e)}')
    
    def execute_single_task(self, task):
        """
        Execute a single task
        """
        if self.emergency_stop_active:
            self.get_logger().warn('Emergency stop active, skipping task execution')
            return False
        
        self.robot_state = RobotState.PROCESSING
        
        try:
            if task['type'] == 'navigation':
                return self.execute_navigation_task(task)
            elif task['type'] == 'manipulation':
                return self.execute_manipulation_task(task)
            elif task['type'] == 'interaction':
                return self.execute_interaction_task(task)
            elif task['type'] == 'unrecognized':
                self.speak_response("I didn't understand that command. Could you please repeat?")
                return False
            else:
                self.get_logger().warn(f'Unknown task type: {task["type"]}')
                return False
        
        except Exception as e:
            self.get_logger().error(f'Error executing task {task}: {str(e)}')
            self.speak_response("I encountered an error while executing your command.")
            return False
        finally:
            self.robot_state = RobotState.IDLE
    
    def execute_navigation_task(self, task):
        """
        Execute navigation-related tasks
        """
        action = task['action']
        
        if action == 'navigate_to':
            target_location = task['target']
            return self.navigate_to_location(target_location)
        elif action == 'move_forward':
            return self.execute_walk_forward(1.0)  # Default 1m forward
        elif action == 'move_backward':
            cmd = Twist()
            cmd.linear.x = -0.3  # Move backward
            self.cmd_vel_publisher.publish(cmd)
            time.sleep(1.0)  # Move for 1 second
            self.stop_robot()
            return True
        elif action == 'turn_left':
            return self.execute_turn(-90)  # Turn 90 degrees left
        elif action == 'turn_right':
            return self.execute_turn(90)  # Turn 90 degrees right
        else:
            self.get_logger().warn(f'Unknown navigation action: {action}')
            return False
    
    def execute_manipulation_task(self, task):
        """
        Execute manipulation-related tasks
        """
        if not self.capabilities['manipulation']:
            self.speak_response("I don't have manipulation capabilities.")
            return False
        
        action = task['action']
        obj = task.get('object', 'unknown')
        
        if action == 'pick_up' or action == 'grasp' or action == 'take':
            self.speak_response(f"Attempting to pick up the {obj}")
            # In a real implementation, this would involve:
            # 1. Locating the object
            # 2. Planning a grasp trajectory
            # 3. Executing the grasp
            # 4. Verifying success
            return True
        elif action == 'place' or action == 'drop':
            self.speak_response(f"Attempting to place the {obj}")
            # In a real implementation, this would involve:
            # 1. Locating a placement position
            # 2. Planning a placement trajectory
            # 3. Executing the placement
            # 4. Releasing the object
            return True
        else:
            self.get_logger().warn(f'Unknown manipulation action: {action}')
            return False
    
    def execute_interaction_task(self, task):
        """
        Execute interaction-related tasks
        """
        action = task['action']
        
        if action == 'speak':
            text = task.get('text', 'Hello')
            self.speak_response(text)
            return True
        elif action == 'greet':
            self.speak_response("Hello! Nice to meet you.")
            return True
        elif action == 'introduce':
            intro_text = "I am an autonomous humanoid robot. I can navigate, understand voice commands, and interact with my environment."
            self.speak_response(intro_text)
            return True
        elif action == 'capabilities':
            cap_text = "I can navigate to locations, understand voice commands, maintain balance, and interact with humans."
            self.speak_response(cap_text)
            return True
        elif action == 'stop':
            self.stop_robot()
            self.speak_response("Stopping all motion.")
            return True
        else:
            self.get_logger().warn(f'Unknown interaction action: {action}')
            return False
    
    def navigate_to_location(self, location):
        """
        Navigate to a specific location
        """
        # This would integrate with navigation stack
        # For this example, we'll simulate navigation to known locations
        
        known_locations = {
            'kitchen': np.array([2.0, 1.0]),
            'living room': np.array([0.0, 0.0]),
            'bedroom': np.array([-1.5, 2.0]),
            'office': np.array([1.0, -1.5])
        }
        
        if location in known_locations:
            target_pos = known_locations[location]
            current_pos = self.current_pose[:2]
            
            # Calculate direction to target
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:  # If not already at location
                self.speak_response(f"Navigating to {location}")
                
                # Simple proportional navigation
                velocity = 0.3  # m/s
                duration = distance / velocity
                
                cmd = Twist()
                cmd.linear.x = velocity
                
                start_time = time.time()
                while time.time() - start_time < duration and not self.emergency_stop_active:
                    self.cmd_vel_publisher.publish(cmd)
                    time.sleep(0.01)
                
                self.stop_robot()
                self.speak_response(f"Arrived at {location}")
                return True
            else:
                self.speak_response(f"I'm already at {location}")
                return True
        else:
            self.speak_response(f"I don't know where {location} is.")
            return False
    
    def stop_robot(self):
        """
        Stop all robot motion
        """
        cmd = Twist()
        self.cmd_vel_publisher.publish(cmd)
    
    def speak_response(self, text):
        """
        Speak a response
        """
        response_msg = String()
        response_msg.data = text
        self.speech_publisher.publish(response_msg)
        self.get_logger().info(f"Speaking: {text}")
    
    def emergency_stop_callback(self, msg):
        """
        Handle emergency stop commands
        """
        if msg.data.lower() in ['stop', 'emergency', 'halt']:
            self.emergency_stop_active = True
            self.stop_robot()
            self.robot_state = RobotState.EMERGENCY_STOP
            self.get_logger().warn('Emergency stop activated!')
        elif msg.data.lower() == 'resume':
            self.emergency_stop_active = False
            self.robot_state = RobotState.IDLE
            self.get_logger().info('Emergency stop cleared, resuming operations')
    
    def monitor_robot_state(self):
        """
        Monitor robot state for safety and balance
        """
        # Check balance state
        if np.linalg.norm(self.current_balance) > 0.1:  # If CoM offset > 10cm
            if self.robot_state != RobotState.BALANCING:
                self.get_logger().warn('Robot is out of balance, initiating correction')
                self.robot_state = RobotState.BALANCING
                self.execute_balance_correction()
                self.robot_state = RobotState.IDLE
        
        # Check for emergency conditions
        if self.emergency_stop_active:
            self.robot_state = RobotState.EMERGENCY_STOP
            self.stop_robot()
    
    def get_robot_capabilities(self):
        """
        Get robot capabilities as a string
        """
        caps = []
        if self.capabilities['navigation']:
            caps.append('navigation')
        if self.capabilities['manipulation']:
            caps.append('manipulation')
        if self.capabilities['voice_interaction']:
            caps.append('voice interaction')
        if self.capabilities['balance_control']:
            caps.append('balance control')
        
        return ', '.join(caps)

def main(args=None):
    rclpy.init(args=args)
    
    robot = AutonomousHumanoidRobot()
    
    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        robot.get_logger().info('Shutting down Autonomous Humanoid Robot')
    finally:
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Assessment and evaluation

Each project should be assessed based on specific criteria:

### Assessment Rubric

| Project | Criteria | Weight | Pass Threshold |
|---------|----------|--------|----------------|
| ROS 2 Package Creation | Code organization, ROS 2 best practices, functionality | 25% | 80% |
| Gazebo Simulation | Realistic physics, proper URDF, working controllers | 25% | 80% |
| Isaac Perception | Visual quality, sensor simulation, integration | 20% | 80% |
| VLA Voice Control | Voice recognition accuracy, command execution, integration | 20% | 80% |
| Final Humanoid Project | Integration of all components, autonomous operation | 10% | 80% |

### Testing Procedures

Each project should undergo the following tests:

1. **Unit Tests**: Test individual components and functions
2. **Integration Tests**: Test how components work together
3. **System Tests**: Test the complete system functionality
4. **Performance Tests**: Measure computational efficiency
5. **Safety Tests**: Verify safe operation under various conditions

## Conclusion

These projects provide hands-on experience with key humanoid robotics concepts. Students progress from basic ROS 2 package creation to complex autonomous behavior implementation. Each project builds on previous knowledge while introducing new concepts and challenges.

The combination of simulation (Gazebo), perception (Isaac), and real-world interaction (VLA, voice control) prepares students for developing complete humanoid robotics applications. The final project integrates all learned concepts into a functional autonomous humanoid system.