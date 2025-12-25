---
title: Introduction to ROS 2
sidebar_position: 1
description: An introduction to ROS 2, its architecture, and fundamental concepts including nodes, topics, services, and actions
---

# Introduction to ROS 2

## What is ROS 2 and why it's important

Robot Operating System 2 (ROS 2) is the next-generation robotics middleware that provides libraries and tools to help software developers create robot applications. Unlike the original ROS, ROS 2 is designed to be production-ready with improved security, real-time capabilities, and better architecture.

### Key Improvements in ROS 2:

- **Production Ready**: Designed for industrial and commercial applications
- **Security**: Built-in security features for safe deployment
- **Real-time Support**: Capabilities for real-time systems
- **Quality of Service**: Configurable reliability and performance options
- **Cross-platform**: Better support for different operating systems
- **DDS-based**: Uses Data Distribution Service (DDS) for communication

### Why ROS 2 is important for humanoid robotics:

- **Hardware Abstraction**: Provides consistent interfaces for different hardware
- **Device Drivers**: Extensive library of drivers for sensors and actuators
- **Ecosystem**: Large community and package repository
- **Standardization**: Common tools and practices across the robotics industry
- **Scalability**: Can handle complex systems with many components

## ROS 2 architecture

ROS 2 has a fundamentally different architecture compared to ROS 1, primarily based on DDS (Data Distribution Service) for communication.

### Core Architecture Components:

#### DDS (Data Distribution Service)
- **Communication Middleware**: Provides publish/subscribe, request/reply communication
- **Language Independent**: Supports multiple programming languages
- **Platform Independent**: Works across different operating systems
- **Quality of Service**: Configurable reliability and performance settings

#### RMW (ROS Middleware)
- **Abstraction Layer**: Hides DDS implementation details
- **Multiple DDS Implementations**: Supports different DDS vendors
- **Switching Capability**: Can switch between DDS implementations without code changes

#### rcl (ROS Client Library)
- **Client Library Interface**: Common interface for all client libraries
- **Implementation Consistency**: Ensures consistent behavior across languages
- **Resource Management**: Handles lifecycle of ROS entities

#### rclcpp/rclpy
- **C++/Python Client Libraries**: Language-specific implementations
- **High-level APIs**: Convenient interfaces for application development
- **Integration**: Seamless integration with existing C++/Python code

## Nodes, Topics, Services, Actions

### Nodes
Nodes are the fundamental building blocks of ROS 2 applications. Each node runs a specific task and communicates with other nodes through messages.

#### Node Characteristics:
- **Process**: Each node runs as a separate process
- **Names**: Identified by unique names within the system
- **Namespace**: Can be organized in namespaces for better management
- **Lifecycle**: Has a lifecycle that can be managed by lifecycle nodes

#### Creating a Node (Python example):
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Node initialization code here

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics
Topics enable publish/subscribe communication between nodes. Multiple nodes can publish to or subscribe to the same topic.

#### Topic Characteristics:
- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Many-to-many**: Multiple publishers and subscribers can use the same topic
- **Typed Messages**: All messages on a topic have the same type
- **Quality of Service**: Configurable delivery guarantees

#### Topic Communication Example:
```python
# Publisher
publisher = self.create_publisher(String, 'topic_name', 10)

# Subscriber
subscriber = self.create_subscription(
    String,
    'topic_name',
    self.callback_function,
    10
)
```

### Services
Services provide request/reply communication between nodes. A client sends a request and waits for a response from a server.

#### Service Characteristics:
- **Synchronous**: Client waits for response from server
- **One-to-one**: One client communicates with one server at a time
- **Request/Response**: Defined message types for request and response
- **Blocking**: Client blocks until response is received

#### Service Example:
```python
# Service server
service = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

# Service client
client = self.create_client(AddTwoInts, 'add_two_ints')
```

### Actions
Actions provide a way to handle long-running tasks with feedback and goal management. They combine features of services and topics.

#### Action Characteristics:
- **Long-running**: Designed for operations that take significant time
- **Feedback**: Provides intermediate feedback during execution
- **Goal Management**: Supports goal cancellation and preemption
- **Multiple States**: Active, succeeded, aborted, canceled

#### Action Example:
```python
# Action server
action_server = ActionServer(
    self,
    Fibonacci,
    'fibonacci',
    self.execute_callback
)

# Action client
action_client = ActionClient(self, Fibonacci, 'fibonacci')
```

## Message passing

Message passing is the core communication mechanism in ROS 2, enabling nodes to exchange information.

### Message Types:
- **Standard Types**: Built-in types like String, Int32, Float64
- **Custom Types**: User-defined message types for specific applications
- **Complex Types**: Messages with nested structures and arrays

### Message Definition (.msg files):
```
# Custom message definition
string name
int32 id
float64[] position  # Array of positions
geometry_msgs/Pose pose  # Nested message
```

### Quality of Service (QoS) Settings:
- **Reliability**: Best effort vs reliable delivery
- **Durability**: Volatile vs transient local
- **History**: Keep last N messages vs keep all
- **Depth**: Size of message queue

## ROS 2 graph

The ROS 2 graph represents the network of nodes and their communication connections.

### Graph Components:
- **Nodes**: Individual processes running ROS 2 code
- **Topics**: Communication channels between nodes
- **Services**: Request/reply connections
- **Actions**: Long-running task connections
- **Parameters**: Configuration values shared between nodes

### Graph Visualization:
- **rqt_graph**: GUI tool for visualizing the ROS graph
- **command line tools**: ros2 node, ros2 topic, ros2 service commands
- **API access**: Programmatic access to graph information

### Command Line Tools:
```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /topic_name

# Call a service
ros2 service call /service_name service_type "request_data"
```

## Conclusion

ROS 2 provides a robust foundation for developing complex robotic systems. Its improved architecture addresses many limitations of ROS 1 and makes it suitable for production environments. Understanding the core concepts of nodes, topics, services, and actions is essential for developing effective robotic applications, particularly for humanoid robots that require coordination between many different components.

## Next Steps

To continue learning about ROS 2 development:

- Continue to [Chapter 5: ROS 2 Development with Python](../part2/chapter5) to learn practical implementation techniques
- Explore [Chapter 6: Robot Description (URDF & XACRO)](../part2/chapter6) to understand how to model robots for ROS