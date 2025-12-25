---
title: NVIDIA Isaac Integration
sidebar_position: 6
description: Integrating NVIDIA Isaac Sim, Isaac ROS, and CUDA-accelerated perception for humanoid robotics
---

# NVIDIA Isaac Integration

## Isaac Sim for humanoid simulation

NVIDIA Isaac Sim is a powerful simulation environment designed for robotics development, particularly for complex systems like humanoid robots. It provides high-fidelity physics simulation, photorealistic rendering, and integration with NVIDIA's GPU-accelerated AI frameworks.

### Introduction to Isaac Sim

Isaac Sim is built on NVIDIA's Omniverse platform and provides:

- **Photorealistic rendering** using RTX technology
- **High-fidelity physics simulation** with PhysX
- **Synthetic data generation** capabilities
- **AI training environments** with domain randomization
- **ROS 2 bridge** for seamless integration with ROS-based workflows
- **Cloud-native simulation** capabilities

### Installation and Setup

```bash
# Install Isaac Sim (typically through Omniverse Launcher)
# Or via pip for Python API
pip install omni.isaac.orbit
```

### Basic Simulation Setup

```python
import omni
import carb
from pxr import Gf, UsdGeom, PhysxSchema
import numpy as np

class HumanoidSimulation:
    def __init__(self):
        # Initialize Isaac Sim components
        self.stage = omni.usd.get_context().get_stage()
        self.world = None
        self.robot = None
        self.physics = None
        
        # Simulation parameters
        self.sim_frequency = 60.0  # Hz
        self.control_frequency = 200.0  # Hz (higher than sim for control)
        self.gravity = -9.81
        
        # Robot parameters
        self.robot_mass = 34.0  # kg (Unitree G1)
        self.com_height = 0.78  # m
        
        # Initialize simulation
        self.setup_scene()
        self.setup_robot()
        self.setup_physics()
    
    def setup_scene(self):
        """
        Setup the simulation scene with ground plane and basic environment
        """
        # Create default prim
        default_prim_path = carb.tokens.get_tokens().defaultPrim
        if not self.stage.GetPrimAtPath("/" + default_prim_path):
            self.stage.DefinePrim("/" + default_prim_path, "Xform")
        
        # Set up default stage
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        
        # Create ground plane
        self.ground_plane = UsdGeom.Mesh.Define(self.stage, "/World/groundPlane")
        self.ground_plane.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
        self.ground_plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        self.ground_plane.CreateFaceVertexCountsAttr([4])
        self.ground_plane.CreateDoubleSidedAttr(True)
        
        # Create collision for ground
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(self.ground_plane.GetPrim())
        collision_api.CreateCollisionEnabledAttr(True)
        
        # Create material for ground
        material_path = "/World/GroundMaterial"
        material = UsdShade.Material.Define(self.stage, material_path)
        
        # Add preview surface shader
        shader = UsdShade.Shader.Define(self.stage, material_path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Set material properties (gray concrete-like)
        shader.CreateInput("diffuseColor", Gf.Vec3f(0.5, 0.5, 0.5)).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        shader.CreateInput("roughness", carb.Float).Set(0.8)
        shader.CreateInput("metallic", carb.Float).Set(0.0)
        
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # Assign material to ground
        rel = self.ground_plane.GetMaterialRel()
        rel.SetTargets([material_path])
    
    def setup_robot(self):
        """
        Setup humanoid robot in the simulation
        """
        # Import or create robot USD
        # For this example, we'll create a simplified humanoid
        robot_path = "/World/HumanoidRobot"
        
        # Create robot root
        robot_root = UsdGeom.Xform.Define(self.stage, robot_path)
        
        # Create torso (main body)
        torso_path = f"{robot_path}/torso"
        torso = UsdGeom.Capsule.Define(self.stage, torso_path)
        torso.GetSizeAttr().Set(0.2)  # Radius
        torso.GetHeightAttr().Set(0.6)  # Height
        torso.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.8))  # Position torso at 0.8m height
        
        # Add rigid body properties
        torso_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(torso.GetPrim())
        torso_rigid.GetMassAttr().Set(self.robot_mass * 0.4)  # 40% of total mass
        
        # Create head
        head_path = f"{robot_path}/head"
        head = UsdGeom.Sphere.Define(self.stage, head_path)
        head.GetRadiusAttr().Set(0.1)
        head.AddTranslateOp().Set(Gf.Vec3f(0, 0, 1.2))  # Position above torso
        
        # Add head to robot hierarchy
        robot_root.GetPrim().GetChildren().append(head.GetPrim())
        
        # Create left leg
        left_hip_path = f"{robot_path}/left_hip"
        left_hip = UsdGeom.Capsule.Define(self.stage, left_hip_path)
        left_hip.GetSizeAttr().Set(0.08)
        left_hip.GetHeightAttr().Set(0.5)
        left_hip.AddTranslateOp().Set(Gf.Vec3f(-0.1, 0, 0.4))
        
        left_knee_path = f"{robot_path}/left_knee"
        left_knee = UsdGeom.Capsule.Define(self.stage, left_knee_path)
        left_knee.GetSizeAttr().Set(0.08)
        left_knee.GetHeightAttr().Set(0.5)
        left_knee.AddTranslateOp().Set(Gf.Vec3f(-0.1, 0, -0.1))
        
        left_ankle_path = f"{robot_path}/left_ankle"
        left_ankle = UsdGeom.Capsule.Define(self.stage, left_ankle_path)
        left_ankle.GetSizeAttr().Set(0.08)
        left_ankle.GetHeightAttr().Set(0.1)
        left_ankle.AddTranslateOp().Set(Gf.Vec3f(-0.1, 0, -0.6))
        
        # Create right leg (similar to left)
        right_hip_path = f"{robot_path}/right_hip"
        right_hip = UsdGeom.Capsule.Define(self.stage, right_hip_path)
        right_hip.GetSizeAttr().Set(0.08)
        right_hip.GetHeightAttr().Set(0.5)
        right_hip.AddTranslateOp().Set(Gf.Vec3f(0.1, 0, 0.4))
        
        right_knee_path = f"{robot_path}/right_knee"
        right_knee = UsdGeom.Capsule.Define(self.stage, right_knee_path)
        right_knee.GetSizeAttr().Set(0.08)
        right_knee.GetHeightAttr().Set(0.5)
        right_knee.AddTranslateOp().Set(Gf.Vec3f(0.1, 0, -0.1))
        
        right_ankle_path = f"{robot_path}/right_ankle"
        right_ankle = UsdGeom.Capsule.Define(self.stage, right_ankle_path)
        right_ankle.GetSizeAttr().Set(0.08)
        right_ankle.GetHeightAttr().Set(0.1)
        right_ankle.AddTranslateOp().Set(Gf.Vec3f(0.1, 0, -0.6))
        
        # Create left arm
        left_shoulder_path = f"{robot_path}/left_shoulder"
        left_shoulder = UsdGeom.Capsule.Define(self.stage, left_shoulder_path)
        left_shoulder.GetSizeAttr().Set(0.06)
        left_shoulder.GetHeightAttr().Set(0.4)
        left_shoulder.AddTranslateOp().Set(Gf.Vec3f(-0.2, 0.2, 0.8))
        
        left_elbow_path = f"{robot_path}/left_elbow"
        left_elbow = UsdGeom.Capsule.Define(self.stage, left_elbow_path)
        left_elbow.GetSizeAttr().Set(0.06)
        left_elbow.GetHeightAttr().Set(0.4)
        left_elbow.AddTranslateOp().Set(Gf.Vec3f(-0.4, 0.2, 0.8))
        
        # Create right arm (similar to left)
        right_shoulder_path = f"{robot_path}/right_shoulder"
        right_shoulder = UsdGeom.Capsule.Define(self.stage, right_shoulder_path)
        right_shoulder.GetSizeAttr().Set(0.06)
        right_shoulder.GetHeightAttr().Set(0.4)
        right_shoulder.AddTranslateOp().Set(Gf.Vec3f(0.2, 0.2, 0.8))
        
        right_elbow_path = f"{robot_path}/right_elbow"
        right_elbow = UsdGeom.Capsule.Define(self.stage, right_elbow_path)
        right_elbow.GetSizeAttr().Set(0.06)
        right_elbow.GetHeightAttr().Set(0.4)
        right_elbow.AddTranslateOp().Set(Gf.Vec3f(0.4, 0.2, 0.8))
        
        # Add collision and rigid body properties
        for part_path in [left_hip_path, left_knee_path, left_ankle_path, 
                         right_hip_path, right_knee_path, right_ankle_path,
                         left_shoulder_path, left_elbow_path,
                         right_shoulder_path, right_elbow_path]:
            part_prim = self.stage.GetPrimAtPath(part_path)
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(part_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(part_prim)
            rigid_api.GetMassAttr().Set(2.0)  # 2kg per limb segment
        
        self.robot = robot_root
    
    def setup_physics(self):
        """
        Setup physics scene and parameters
        """
        # Create physics scene
        scene_path = "/World/PhysicsScene"
        scene = PhysicsSchema.PhysicsScene.Define(self.stage, scene_path)
        
        # Set gravity
        scene.CreateGravityAttr().Set(Gf.Vec3f(0.0, 0.0, self.gravity))
        
        # Set simulation parameters
        scene.CreateTimestepAttr().Set(1.0 / self.sim_frequency)
        scene.CreateEnableCCDAttr().Set(True)  # Enable continuous collision detection
        
        # Create articulation for robot (if using joints)
        # This would define the kinematic structure of the humanoid
        self.setup_articulation()
    
    def setup_articulation(self):
        """
        Setup articulation (kinematic structure) for the humanoid
        """
        # Create articulation root for the robot
        articulation_root_path = f"{self.robot.GetPath()}/articulation_root"
        articulation_root = PhysxSchema.PhysxArticulationRootAPI.Apply(self.robot.GetPrim())
        
        # Create joints between body parts (simplified)
        # Hip joints
        self.create_revolute_joint(
            f"{self.robot.GetPath()}/left_hip_joint",
            f"{self.robot.GetPath()}/torso",
            f"{self.robot.GetPath()}/left_hip",
            (0, 0, 0.4),  # Hip position
            (1, 0, 0)  # Axis of rotation (hip pitch)
        )
        
        self.create_revolute_joint(
            f"{self.robot.GetPath()}/right_hip_joint",
            f"{self.robot.GetPath()}/torso",
            f"{self.robot.GetPath()}/right_hip",
            (0, 0, 0.4),
            (1, 0, 0)
        )
        
        # Knee joints
        self.create_revolute_joint(
            f"{self.robot.GetPath()}/left_knee_joint",
            f"{self.robot.GetPath()}/left_hip",
            f"{self.robot.GetPath()}/left_knee",
            (0, 0, -0.25),  # Mid-thigh position
            (1, 0, 0)  # Knee pitch axis
        )
        
        self.create_revolute_joint(
            f"{self.robot.GetPath()}/right_knee_joint",
            f"{self.robot.GetPath()}/right_hip",
            f"{self.robot.GetPath()}/right_knee",
            (0, 0, -0.25),
            (1, 0, 0)
        )
    
    def create_revolute_joint(self, joint_path, parent_path, child_path, position, axis):
        """
        Create a revolute joint between two bodies
        """
        joint = PhysxSchema.PhysxRevoluteJoint.Define(self.stage, joint_path)
        
        # Set joint properties
        joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        
        # Set joint position and axis
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        
        # Set rotation axis
        joint.CreateAxisAttr().Set(Gf.Vec3f(*axis))
        
        # Set joint limits (if applicable)
        joint.CreateLowerLimitAttr().Set(-2.0)  # radians
        joint.CreateUpperLimitAttr().Set(0.5)   # radians
        
        # Set drive properties for actuation
        joint.CreateDriveTypeAttr().Set("force")
        joint.CreateMaxJointVelAttr().Set(10.0)  # rad/s
        joint.CreateMaxJointForceAttr().Set(200.0)  # N-m
    
    def run_simulation(self, duration=10.0):
        """
        Run the simulation for a specified duration
        """
        # Get simulation timeline
        timeline = omni.timeline.get_timeline_interface()
        
        # Start simulation
        timeline.play()
        
        # Run for specified duration
        start_time = timeline.get_current_time()
        while timeline.get_current_time() < start_time + duration:
            # Update simulation
            timeline.update()
            
            # Apply control inputs periodically (higher frequency than sim)
            if int(timeline.get_current_time() * self.control_frequency) % 1 == 0:
                self.apply_control_inputs()
            
            # Yield control back to simulator
            carb.app.acquire_app().update()
        
        # Stop simulation
        timeline.stop()
    
    def apply_control_inputs(self):
        """
        Apply control inputs to robot joints
        This would interface with your actual controller
        """
        # Example: Apply simple walking pattern to leg joints
        current_time = omni.timeline.get_timeline_interface().get_current_time()
        
        # Create oscillating walking pattern
        phase = current_time * 2 * np.pi * 0.5  # 0.5 Hz walking
        
        # Apply to left hip and knee (simplified)
        left_hip_joint = self.stage.GetPrimAtPath(f"{self.robot.GetPath()}/left_hip_joint")
        left_knee_joint = self.stage.GetPrimAtPath(f"{self.robot.GetPath()}/left_knee_joint")
        
        if left_hip_joint and left_knee_joint:
            # Apply walking gait pattern
            hip_command = 0.1 * np.sin(phase)
            knee_command = 0.2 * np.cos(phase)
            
            # In a real implementation, you would set joint drives
            # This is a simplified representation
            print(f"Applying commands: hip={hip_command:.3f}, knee={knee_command:.3f}")
    
    def get_robot_state(self):
        """
        Get current robot state from simulation
        """
        # This would extract position, velocity, and other state information
        # from the simulated robot
        state = {
            'joint_positions': {},
            'joint_velocities': {},
            'com_position': np.array([0.0, 0.0, 0.8]),
            'imu_data': {
                'orientation': [0, 0, 0, 1],  # w, x, y, z quaternion
                'angular_velocity': [0, 0, 0],
                'linear_acceleration': [0, 0, self.gravity]
            }
        }
        
        return state

# Example usage
sim = HumanoidSimulation()
sim.run_simulation(duration=5.0)
```

### Advanced Isaac Sim Features

```python
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omi.isaac.core.robots import Robot
from omi.isaac.core.articulations import Articulation
from omi.isaac.core.utils.nucleus import get_assets_root_path
from omi.isaac.core.utils.viewports import set_camera_view
import numpy as np

class AdvancedHumanoidSim:
    def __init__(self):
        # Create Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        
        # Robot parameters
        self.robot_usd_path = "/path/to/humanoid_robot.usd"  # Path to robot USD file
        self.robot_position = np.array([0.0, 0.0, 0.8])  # Initial position
        self.robot_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Initial orientation (quaternion)
        
        # Physics parameters
        self.sim_frequency = 60.0
        self.control_frequency = 200.0
        self.gravity = -9.81
        
        # Initialize the world
        self.setup_environment()
        self.load_robot()
        self.setup_sensors()
        
    def setup_environment(self):
        """
        Setup advanced simulation environment with objects and scenarios
        """
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add objects for interaction
        self.add_interactable_objects()
        
        # Set up lighting
        self.setup_environmental_lighting()
        
        # Configure physics scene
        self.configure_physics_settings()
    
    def add_interactable_objects(self):
        """
        Add objects that the humanoid can interact with
        """
        # Add various objects for testing manipulation
        prim_utils.create_primitive(
            prim_path="/World/Box",
            primitive_props={"prim_type": "Cube", "scale": np.array([0.1, 0.1, 0.1])},
            translation=np.array([0.5, 0.0, 0.1]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        # Add cylinder
        prim_utils.create_primitive(
            prim_path="/World/Cylinder",
            primitive_props={"prim_type": "Cylinder", "scale": np.array([0.05, 0.05, 0.15])},
            translation=np.array([0.7, 0.2, 0.075]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        # Add sphere
        prim_utils.create_primitive(
            prim_path="/World/Sphere",
            primitive_props={"prim_type": "Sphere", "scale": np.array([0.07, 0.07, 0.07])},
            translation=np.array([0.6, -0.2, 0.07]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        # Add furniture for navigation testing
        prim_utils.create_primitive(
            prim_path="/World/Table",
            primitive_props={"prim_type": "Cuboid", "scale": np.array([0.8, 0.6, 0.8])},
            translation=np.array([1.0, 0.0, 0.4]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    
    def setup_environmental_lighting(self):
        """
        Configure advanced lighting for photorealistic rendering
        """
        # Add dome light for environment lighting
        dome_light = prim_utils.create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            translation=np.array([0.0, 0.0, 0.0]),
            attributes={"color": np.array([0.9, 0.9, 1.0]), "intensity": 3000.0}
        )
        
        # Add distant light for more realistic shadows
        distant_light = prim_utils.create_prim(
            prim_path="/World/DistantLight",
            prim_type="DistantLight",
            translation=np.array([0.0, 0.0, 10.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            attributes={
                "color": np.array([1.0, 0.95, 0.9]),
                "intensity": 400.0,
                "angle": 0.5
            }
        )
        
        # Add textures and materials for realism
        self.add_realistic_materials()
    
    def add_realistic_materials(self):
        """
        Add realistic materials to objects for better visual fidelity
        """
        # Create realistic materials using MDL (Material Definition Language)
        from omni.isaac.core.materials import OmniPBR
    
    def configure_physics_settings(self):
        """
        Configure advanced physics settings
        """
        # Get physics scene
        scene = self.world.scene._scene
        physics_scene = scene.get_physics_scene_option()
        
        # Set advanced physics parameters
        physics_scene.set_gravity(self.gravity)
        physics_scene.set_timestep(1.0 / self.sim_frequency)
        physics_scene.enable_ccd(True)  # Continuous collision detection
        
        # Set solver parameters
        physics_scene.set_position_iteration_count(8)
        physics_scene.set_velocity_iteration_count(1)
        
        # Set broadphase type (SAP or MBP)
        physics_scene.set_broadphase_type("MBP")  # Multi-Box Pruning for better performance
        
    def load_robot(self):
        """
        Load the humanoid robot model into the simulation
        """
        # Add robot to the world
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/HumanoidRobot",
                name="humanoid_robot",
                usd_path=self.robot_usd_path,
                position=self.robot_position,
                orientation=self.robot_orientation
            )
        )
        
        # Wait for world to initialize
        self.world.reset()
    
    def setup_sensors(self):
        """
        Setup various sensors for the humanoid robot
        """
        # Add IMU sensor
        self.add_imu_sensor()
        
        # Add camera sensors
        self.add_camera_sensors()
        
        # Add force/torque sensors at feet
        self.add_force_torque_sensors()
        
        # Add LiDAR sensor (if needed)
        self.add_lidar_sensor()
    
    def add_imu_sensor(self):
        """
        Add IMU sensor to the robot's torso
        """
        from omni.isaac.sensor import Imu
        
        self.imu_sensor = Imu(
            prim_path=f"{self.robot.prim_path}/torso/imu",
            frequency=100  # 100 Hz
        )
    
    def add_camera_sensors(self):
        """
        Add camera sensors to the robot (head, chest, etc.)
        """
        from omni.isaac.sensor import Camera
        
        # Head camera (forward facing)
        self.head_camera = Camera(
            prim_path=f"{self.robot.prim_path}/head/head_camera",
            name="head_camera",
            position=np.array([0.1, 0.0, 0.05]),  # Slightly forward and up from head center
            frequency=30  # 30 Hz
        )
        
        # Chest camera (downward facing for footstep planning)
        self.chest_camera = Camera(
            prim_path=f"{self.robot.prim_path}/torso/chest_camera",
            name="chest_camera",
            position=np.array([0.0, 0.0, -0.2]),  # Looking down from chest
            frequency=30
        )
        
        # Set camera properties
        self.head_camera.set_resolution((640, 480))
        self.chest_camera.set_resolution((640, 480))
    
    def add_force_torque_sensors(self):
        """
        Add force/torque sensors to robot feet for balance control
        """
        from omni.isaac.sensor import ContactSensor
        
        # Left foot sensor
        self.left_foot_sensor = ContactSensor(
            prim_path=f"{self.robot.prim_path}/left_foot/contact_sensor",
            name="left_foot_contact",
            min_threshold=1.0,
            max_threshold=1000.0
        )
        
        # Right foot sensor
        self.right_foot_sensor = ContactSensor(
            prim_path=f"{self.robot.prim_path}/right_foot/contact_sensor",
            name="right_foot_contact",
            min_threshold=1.0,
            max_threshold=1000.0
        )
    
    def add_lidar_sensor(self):
        """
        Add LiDAR sensor for environment perception
        """
        from omni.isaac.sensor import LidarRtx
        
        self.lidar_sensor = LidarRtx(
            prim_path=f"{self.robot.prim_path}/head/lidar",
            name="robot_lidar",
            translation=np.array([0.0, 0.0, 0.1]),  # Above head
            config="45degree",
            rotation_rate=10  # 10 Hz rotation
        )
    
    def run_advanced_simulation(self, duration=10.0):
        """
        Run advanced simulation with sensor data collection and control
        """
        # Reset world
        self.world.reset()
        
        # Simulation loop
        for step in range(int(duration * self.sim_frequency)):
            # Step the world
            self.world.step(render=True)
            
            # Get sensor data
            if step % int(self.sim_frequency / 30) == 0:  # 30 Hz sensor reading
                self.read_sensors()
            
            # Apply control at higher frequency
            if step % int(self.sim_frequency / self.control_frequency) == 0:
                self.apply_advanced_control()
            
            # Occasionally print status
            if step % int(self.sim_frequency) == 0:  # Every second
                robot_pos, robot_orn = self.robot.get_world_pose()
                print(f"Time: {step/self.sim_frequency:.1f}s, Robot position: {robot_pos}")
    
    def read_sensors(self):
        """
        Read data from all sensors
        """
        # Get IMU data
        imu_data = self.imu_sensor.get_measurements()
        
        # Get camera data
        head_rgb = self.head_camera.get_rgb()
        head_depth = self.head_camera.get_depth()
        
        chest_rgb = self.chest_camera.get_rgb()
        chest_depth = self.chest_camera.get_depth()
        
        # Get contact sensor data
        left_contact = self.left_foot_sensor.get_measurements()
        right_contact = self.right_foot_sensor.get_measurements()
        
        # Get LiDAR data (if available)
        if hasattr(self, 'lidar_sensor'):
            lidar_data = self.lidar_sensor.get_point_cloud()
        
        # Store sensor data for control algorithms
        self.sensor_data = {
            'imu': imu_data,
            'head_camera': {'rgb': head_rgb, 'depth': head_depth},
            'chest_camera': {'rgb': chest_rgb, 'depth': chest_depth},
            'left_foot_contact': left_contact,
            'right_foot_contact': right_contact,
            'lidar': lidar_data if 'lidar_data' in locals() else None
        }
    
    def apply_advanced_control(self):
        """
        Apply advanced control algorithms to the robot
        """
        # Get current robot state
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        
        # Calculate control commands based on sensor data
        control_commands = self.calculate_control_commands(
            self.sensor_data,
            joint_positions,
            joint_velocities
        )
        
        # Apply commands to robot
        self.robot.apply_actions(control_commands)
    
    def calculate_control_commands(self, sensor_data, joint_pos, joint_vel):
        """
        Calculate control commands based on sensor data and control algorithms
        """
        # This would implement your specific control algorithm
        # For example: balance control using ZMP, walking pattern generation, etc.
        
        # Placeholder: simple joint position control
        desired_positions = joint_pos  # In real implementation, this would be calculated
        
        # Calculate position-based control commands
        commands = {
            'positions': desired_positions,
            'velocities': np.zeros_like(joint_pos),
            'efforts': np.zeros_like(joint_pos)
        }
        
        return commands
    
    def get_robot_state(self):
        """
        Get comprehensive robot state
        """
        # Get joint states
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        joint_efforts = self.robot.get_joint_efforts()
        
        # Get base state
        base_position, base_orientation = self.robot.get_world_pose()
        base_linear_vel, base_angular_vel = self.robot.get_world_velocities()
        
        # Get CoM state (approximated)
        com_position = self.approximate_com_position(joint_positions)
        
        # Get sensor data if available
        sensor_readings = getattr(self, 'sensor_data', {})
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'joint_efforts': joint_efforts,
            'base_position': base_position,
            'base_orientation': base_orientation,
            'base_linear_velocity': base_linear_vel,
            'base_angular_velocity': base_angular_vel,
            'com_position': com_position,
            'sensor_data': sensor_readings
        }
    
    def approximate_com_position(self, joint_positions):
        """
        Approximate center of mass position based on joint configuration
        """
        # This is a simplified approximation
        # In reality, CoM calculation requires detailed mass distribution
        return np.array([0.0, 0.0, 0.8])  # Approximate CoM height for humanoid

# Example usage
advanced_sim = AdvancedHumanoidSim()

# Run simulation
advanced_sim.run_advanced_simulation(duration=10.0)

# Get final robot state
final_state = advanced_sim.get_robot_state()
print(f"Final robot position: {final_state['base_position']}")
print(f"Final CoM position: {final_state['com_position']}")
```

## Isaac ROS for perception

Isaac ROS bridges the gap between Isaac Sim and the ROS 2 ecosystem, providing GPU-accelerated perception nodes that leverage NVIDIA's hardware acceleration.

### Isaac ROS Architecture

Isaac ROS provides a collection of hardware-accelerated perception nodes that run as ROS 2 packages. These nodes leverage CUDA, TensorRT, and other NVIDIA technologies for efficient processing.

### Installation and Setup

```bash
# Add NVIDIA package repositories
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
sudo apt update
sudo apt install -y gnupg2 curl

# Add NVIDIA's public GPG key
curl -sSL https://developer.download.nvidia.com/installers/add-apt-repository.sh | sudo bash

# Install Isaac ROS packages
sudo apt update
sudo apt install -y ros-humble-isaac-ros-common
sudo apt install -y ros-humble-isaac-ros-perception
sudo apt install -y ros-humble-isaac-ros-nav2
```

### Isaac ROS Perception Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import Header
import message_filters

class IsaacROSPipeline(Node):
    def __init__(self):
        super().__init__('isaac_ros_pipeline')
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Create subscribers for camera and depth data
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_rect_color')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_rect_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/rgb/camera_info')
        
        # Synchronize topics
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], 
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.camera_callback)
        
        # Create publishers for processed data
        self.detection_pub = self.create_publisher(Detection2DArray, '/isaac_ros/detections', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/isaac_ros/pointcloud', 10)
        self.computed_depth_pub = self.create_publisher(Image, '/isaac_ros/computed_depth', 10)
        
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
        
        self.get_logger().info('Isaac ROS Perception Pipeline initialized')
    
    def initialize_isaac_perception_nodes(self):
        """
        Initialize Isaac ROS perception nodes
        In practice, these would be launched separately via launch files
        """
        # Placeholder for Isaac ROS node initialization
        # In a real implementation, these would be actual Isaac ROS packages:
        # - Isaac ROS Image Pipeline
        # - Isaac ROS Detection Pipeline  
        # - Isaac ROS Depth Pipeline
        # - Isaac ROS PointCloud Pipeline
        
        # For this example, we'll simulate the functionality
        self.isaac_initialized = True
        
        # Load Isaac ROS compatible models
        try:
            import torch
            # Example: Load Isaac ROS detection model (placeholder)
            self.detection_model = self.load_isaac_detection_model()
            self.get_logger().info('Isaac ROS detection model loaded')
        except ImportError:
            self.get_logger().warn('PyTorch not available, skipping Isaac ROS model loading')
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
            model = torch.hub.load('nvidia/isaac-ros-dev', 'isaac_ros_detectnet', 
                                  pretrained=True, trust_repo=True)
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load Isaac ROS model: {str(e)}')
            return None
    
    def camera_callback(self, rgb_msg, depth_msg, info_msg):
        """
        Process synchronized camera data using Isaac ROS pipeline
        """
        try:
            # Convert ROS images to OpenCV format
            rgb_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            
            # Apply Isaac ROS perception pipeline
            detections = self.process_with_isaac_detection(rgb_image)
            pointcloud = self.process_with_isaac_depth(depth_image, info_msg)
            
            # Publish results
            if detections:
                self.publish_detections(detections, rgb_msg.header)
            
            if pointcloud:
                self.publish_pointcloud(pointcloud, rgb_msg.header)
                
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS pipeline: {str(e)}')
    
    def process_with_isaac_detection(self, image):
        """
        Process image with Isaac ROS detection pipeline
        """
        if self.detection_model is None:
            return None
        
        try:
            import torch
            
            # Preprocess image for Isaac ROS model
            input_tensor = self.preprocess_for_isaac_model(image)
            
            # Run inference using Isaac ROS model (with TensorRT acceleration if enabled)
            with torch.no_grad():
                if self.use_tensor_rt:
                    # In practice, Isaac ROS uses TensorRT for acceleration
                    # This is a simplified representation
                    detections = self.detection_model(input_tensor)
                else:
                    detections = self.detection_model(input_tensor)
            
            # Post-process detections
            processed_detections = self.post_process_isaac_detections(detections, image.shape)
            
            return processed_detections
            
        except Exception as e:
            self.get_logger().error(f'Error in Isaac ROS detection: {str(e)}')
            return None
    
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
        import torch
        input_tensor = torch.from_numpy(input_tensor).to(torch.float32)
        
        return input_tensor
    
    def post_process_isaac_detections(self, raw_detections, image_shape):
        """
        Post-process Isaac ROS detection outputs
        """
        # This would convert Isaac ROS model outputs to standard ROS messages
        # In practice, Isaac ROS nodes handle this conversion internally
        
        # Placeholder implementation
        height, width = image_shape[:2]
        
        # Example: convert raw detections to Detection2DArray
        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_rgb_optical_frame'
        
        # For demonstration, create some mock detections
        # In reality, this would come from the Isaac ROS model output
        for i in range(3):  # Example: 3 detections
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
            hypothesis.id = np.random.choice(['person', 'chair', 'cup', 'bottle'])
            hypothesis.score = np.random.uniform(0.6, 0.95)
            detection.results.append(hypothesis)
            
            detections.detections.append(detection)
        
        return detections
    
    def process_with_isaac_depth(self, depth_image, camera_info):
        """
        Process depth image with Isaac ROS pipeline
        """
        try:
            # In practice, Isaac ROS would use hardware-accelerated depth processing
            # This is a simplified example
            
            # Use Isaac ROS depth processing techniques:
            # 1. Depth filtering and denoising
            # 2. Point cloud generation
            # 3. Surface normal computation
            
            # Apply depth filtering (simplified)
            filtered_depth = self.filter_depth_image(depth_image)
            
            # Generate point cloud from depth
            pointcloud = self.depth_to_pointcloud(filtered_depth, camera_info)
            
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
    
    def depth_to_pointcloud(self, depth_image, camera_info):
        """
        Convert depth image to point cloud using camera parameters
        """
        # Extract camera parameters
        fx = camera_info.k[0]  # Focal length x
        fy = camera_info.k[4]  # Focal length y
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y
        
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
        header.frame_id = camera_info.header.frame_id
        
        # Create point cloud
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        pointcloud_msg = point_cloud2.create_cloud(header, fields, valid_points)
        
        return pointcloud_msg
    
    def publish_detections(self, detections, header):
        """
        Publish detection results
        """
        # Set appropriate header
        detections.header = header
        self.detection_publisher.publish(detections)
    
    def publish_pointcloud(self, pointcloud, header):
        """
        Publish point cloud results
        """
        # Set appropriate header
        pointcloud.header = header
        self.pointcloud_publisher.publish(pointcloud)

def main(args=None):
    rclpy.init(args=args)
    
    node = IsaacROSPipeline()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Isaac ROS Pipeline')
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
                {'enable_profiling': False}
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

## GPU-accelerated perception

GPU acceleration is crucial for real-time perception in humanoid robots, especially for complex tasks like object detection, segmentation, and 3D reconstruction.

### CUDA-based Perception Pipeline

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

class GPUPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('gpu_perception_pipeline')
        
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
            Detection2DArray, '/gpu_detections', 10
        )
        
        self.feature_pub = self.create_publisher(
            PointStamped, '/gpu_features', 10
        )
        
        # Initialize perception models
        self.initialize_models()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        
        self.get_logger().info('GPU Perception Pipeline initialized')
    
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
        # Set header and publish
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
    
    node = GPUPerceptionPipeline()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down GPU Perception Pipeline')
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
class OptimizedGPUPerception(GPUPerceptionPipeline):
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
    
    node = OptimizedGPUPerception()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Optimized GPU Perception Pipeline')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with humanoid control

Integrating perception with humanoid control requires careful coordination between the perception and control systems to ensure that the robot can react appropriately to visual information.

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

class PerceptionControlInterface(Node):
    def __init__(self):
        super().__init__('perception_control_interface')
        
        # Initialize perception and control components
        self.perception_pipeline = GPUPerceptionPipeline()
        self.balance_controller = None  # Will be initialized separately
        self.walk_controller = None     # Will be initialized separately
        
        # Create subscribers and publishers
        self.object_sub = self.create_subscription(
            Detection2DArray, '/gpu_detections', self.object_detection_callback, 10
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
        
        self.get_logger().info('Perception-Control Interface initialized')
    
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
        
        self.get_logger().info(f'Avoiding person {distance:.2f}m away by stepping {direction}')
    
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
            # No recent detections - continue with default behavior
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
    
    node = PerceptionControlInterface()
    
    try:
        # Run the perception-control loop in a separate thread
        control_thread = threading.Thread(target=node.run_perception_control_loop)
        control_thread.start()
        
        # Spin the node to handle callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Perception-Control Interface')
        node.is_active = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Conclusion

NVIDIA Isaac provides a powerful platform for developing perception and control systems for humanoid robots. Through Isaac Sim, developers can create realistic simulation environments to test and train their robots. Isaac ROS bridges the gap between the simulation and real-world deployment with hardware-accelerated perception nodes. GPU-accelerated perception enables real-time processing of sensor data, which is crucial for responsive humanoid behavior. The integration with humanoid control systems allows robots to react appropriately to their environment based on visual perception.

The combination of these technologies enables the development of sophisticated humanoid robots that can perceive and interact with their environment effectively, forming the basis for more autonomous and capable humanoid systems.