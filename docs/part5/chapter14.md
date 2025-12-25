---
title: Whole Body Control
sidebar_position: 3
description: Advanced control techniques for humanoid robots including center of mass control, zero moment point control, compliance control, and multi-objective optimization
---

# Whole Body Control

## Center of mass control

Whole body control for humanoid robots is a sophisticated approach to managing the complex dynamics of a multi-link system with many degrees of freedom. At its core is the control of the center of mass (CoM), which is crucial for maintaining balance and executing stable movements.

The center of mass represents the average position of all the mass in the robot's body. For a humanoid robot with mass distributed across multiple links (torso, arms, legs, head), the CoM position is calculated as:

$$
CoM = \frac{\sum_{i} m_i \cdot p_i}{\sum_{i} m_i}
$$

Where $ m_i $ is the mass of link $ i $ and $ p_i $ is the position of link $ i $'s center of mass.

### CoM Estimation

Accurate estimation of the CoM is essential for whole body control. This typically involves:

1. **Kinematic State Estimation**: Determining the position and orientation of all links
2. **Mass Distribution Modeling**: Accurate knowledge of mass and center of mass for each link
3. **Sensor Fusion**: Combining data from IMUs, joint encoders, and sometimes force/torque sensors

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class CenterOfMassEstimator:
    def __init__(self, robot_model):
        """
        Initialize CoM estimator with robot model
        robot_model: Contains link masses, geometries, and kinematic structure
        """
        self.robot_model = robot_model
        self.link_properties = self._load_link_properties()
        
    def _load_link_properties(self):
        """
        Load mass and CoM properties for each link
        """
        properties = {}
        for link_name in self.robot_model.links:
            link = self.robot_model.links[link_name]
            properties[link_name] = {
                'mass': link.mass,
                'relative_com': np.array(link.com_offset),  # CoM offset from joint frame
            }
        return properties
    
    def estimate_com(self, joint_positions, base_pose):
        """
        Estimate center of mass position in world frame
        joint_positions: Dictionary mapping joint names to angles
        base_pose: [x, y, z, qx, qy, qz, qw] - base position and orientation
        """
        total_mass = 0.0
        weighted_com_sum = np.zeros(3)
        
        # Extract base position and orientation
        base_pos = np.array(base_pose[:3])
        base_quat = np.array(base_pose[3:])
        base_rot = R.from_quat(base_quat).as_matrix()
        
        # Calculate CoM of each link and accumulate
        for link_name, properties in self.link_properties.items():
            # Get link pose in world frame
            link_pose = self.robot_model.get_link_pose(link_name, joint_positions, base_pose)
            link_pos = link_pose[:3, 3]
            link_rot = link_pose[:3, :3]
            
            # Calculate CoM position in link frame
            local_com = properties['relative_com']
            
            # Transform to world frame
            world_com = link_pos + link_rot @ local_com
            
            # Accumulate weighted contribution
            link_mass = properties['mass']
            weighted_com_sum += link_mass * world_com
            total_mass += link_mass
        
        if total_mass > 0:
            com_position = weighted_com_sum / total_mass
        else:
            com_position = np.zeros(3)
        
        return com_position, total_mass
    
    def estimate_com_velocity(self, joint_positions, joint_velocities, base_pose, base_twist):
        """
        Estimate CoM velocity using analytical differentiation
        """
        total_mass = 0.0
        weighted_com_vel_sum = np.zeros(3)
        
        # Extract base twist
        base_linear_vel = np.array(base_twist[:3])
        base_angular_vel = np.array(base_twist[3:])
        
        # Calculate CoM velocity for each link
        for link_name, properties in self.link_properties.items():
            link_jacobian = self.robot_model.get_link_jacobian(link_name, joint_positions, base_pose)
            
            # Calculate velocity of link origin
            link_origin_vel = self._calculate_link_origin_velocity(
                link_jacobian, joint_positions, joint_velocities, base_linear_vel, base_angular_vel
            )
            
            # Calculate CoM offset velocity due to rotation
            link_pose = self.robot_model.get_link_pose(link_name, joint_positions, base_pose)
            link_rot = link_pose[:3, :3]
            local_com = properties['relative_com']
            com_offset_vel = link_angular_vel.cross(link_rot @ local_com)
            
            # Total CoM velocity in world frame
            link_com_vel = link_origin_vel + com_offset_vel
            
            # Accumulate weighted contribution
            link_mass = properties['mass']
            weighted_com_vel_sum += link_mass * link_com_vel
            total_mass += link_mass
        
        if total_mass > 0:
            com_velocity = weighted_com_vel_sum / total_mass
        else:
            com_velocity = np.zeros(3)
        
        return com_velocity
    
    def estimate_com_acceleration(self, joint_positions, joint_velocities, joint_accelerations, 
                                 base_pose, base_twist, base_acceleration):
        """
        Estimate CoM acceleration using second-order differentiation
        """
        total_mass = 0.0
        weighted_com_acc_sum = np.zeros(3)
        
        # Calculate CoM acceleration for each link
        for link_name, properties in self.link_properties.items():
            link_jac = self.robot_model.get_link_jacobian(link_name, joint_positions, base_pose)
            link_jac_dot = self.robot_model.get_link_jacobian_derivative(
                link_name, joint_positions, joint_velocities, base_pose, base_twist
            )
            
            # Calculate CoM acceleration
            joint_part = link_jac_dot @ np.array(list(joint_velocities.values()))
            acceleration_part = link_jac @ np.array(list(joint_accelerations.values()))
            base_part = self._calculate_base_contribution_to_acceleration(
                link_jac, base_twist, base_acceleration
            )
            
            link_com_acc = joint_part + acceleration_part + base_part
            
            # Accumulate weighted contribution
            link_mass = properties['mass']
            weighted_com_acc_sum += link_mass * link_com_acc
            total_mass += link_mass
        
        if total_mass > 0:
            com_acceleration = weighted_com_acc_sum / total_mass
        else:
            com_acceleration = np.zeros(3)
        
        return com_acceleration

# Example usage
# com_estimator = CenterOfMassEstimator(robot_model)
# com_pos, total_mass = com_estimator.estimate_com(joint_positions, base_pose)
# com_vel = com_estimator.estimate_com_velocity(joint_positions, joint_velocities, base_pose, base_twist)
```

### CoM Control Strategies

Several strategies exist for controlling the center of mass:

#### 1. Inverted Pendulum Model

The simplest model treats the robot as an inverted pendulum:

$$ \ddot{x} = \frac{g}{h}(x - x_{ZMP}) $$

Where $ h $ is the CoM height, $ g $ is gravity, and $ x_{ZMP} $ is the Zero Moment Point.

```python
class InvertedPendulumController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.pendulum_omega = np.sqrt(gravity / com_height)
        
        # Control gains
        self.kp = self.pendulum_omega**2  # Proportional gain
        self.kd = 2 * self.pendulum_omega  # Critical damping
    
    def calculate_zmp_from_com(self, com_pos, com_acc):
        """
        Calculate ZMP from CoM position and acceleration
        """
        zmp_x = com_pos[0] - (self.com_height / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.gravity) * com_acc[1]
        
        return np.array([zmp_x, zmp_y])
    
    def calculate_com_from_zmp(self, zmp_pos, com_pos, com_vel, dt):
        """
        Calculate next CoM position based on desired ZMP
        """
        # Inverted pendulum dynamics
        com_acc_x = self.gravity / self.com_height * (com_pos[0] - zmp_pos[0])
        com_acc_y = self.gravity / self.com_height * (com_pos[1] - zmp_pos[1])
        
        # Integrate to get velocity and position
        new_com_vel_x = com_vel[0] + com_acc_x * dt
        new_com_vel_y = com_vel[1] + com_acc_y * dt
        
        new_com_x = com_pos[0] + new_com_vel_x * dt
        new_com_y = com_pos[1] + new_com_vel_y * dt
        
        return np.array([new_com_x, new_com_y, com_pos[2]]), np.array([new_com_vel_x, new_com_vel_y, com_vel[2]]), np.array([com_acc_x, com_acc_y, 0])
```

#### 2. Linear Inverted Pendulum Model (LIPM)

The Linear Inverted Pendulum Model simplifies the dynamics by assuming constant CoM height:

$$ \ddot{r}_{CoM} = \omega^2(r_{CoM} - r_{ZMP}) $$

Where $ \omega = \sqrt{\frac{g}{h}} $ and $ h $ is the constant CoM height.

```python
class LinearInvertedPendulumController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        self.omega_sq = self.omega ** 2
        
        # For trajectory planning
        self.preview_control_horizon = 2.0  # seconds
        self.dt = 0.01  # 100Hz control rate
    
    def compute_analytical_solution(self, initial_com, initial_com_vel, zmp_trajectory):
        """
        Compute CoM trajectory using analytical solution of LIPM
        """
        omega = self.omega
        t_values = np.linspace(0, len(zmp_trajectory) * self.dt, len(zmp_trajectory))
        
        # Analytical solution for LIPM:
        # r_com(t) = r_zmp + e^(wt)(A) + e^(-wt)(B)
        # where A and B are determined by initial conditions
        
        # Initial conditions
        r0 = initial_com[:2]  # x, y components only
        v0 = initial_com_vel[:2]
        rzmp = zmp_trajectory[0]  # Initial ZMP
        
        # Calculate constants A and B
        A = 0.5 * ((r0 - rzmp) + (v0 / omega))
        B = 0.5 * ((r0 - rzmp) - (v0 / omega))
        
        com_trajectory = []
        for i, t in enumerate(t_values):
            if i < len(zmp_trajectory):
                current_rzmp = zmp_trajectory[i]
                # Note: This is simplified - in reality ZMP changes over time
                com_xy = current_rzmp + np.exp(omega*t) * A + np.exp(-omega*t) * B
                com_pos = np.array([com_xy[0], com_xy[1], initial_com[2]])  # Keep z constant
                com_trajectory.append(com_pos)
        
        return np.array(com_trajectory)
    
    def compute_discrete_solution(self, initial_com, initial_com_vel, zmp_sequence, dt):
        """
        Compute CoM trajectory using discrete-time LIPM
        """
        # Discrete-time solution of LIPM:
        # r_com[k+1] = a*r_com[k] + b*r_com[k-1] + g*zmp[k]
        # where a = (2 + w²dt²)/(1 + w²dt²), b = -(1 - w²dt²)/(1 + w²dt²), g = w²dt²/(1 + w²dt²)
        
        omega_sq_dt_sq = self.omega_sq * dt * dt
        denominator = 1 + omega_sq_dt_sq
        
        a = (2 + omega_sq_dt_sq) / denominator
        b = -(1 - omega_sq_dt_sq) / denominator
        g = omega_sq_dt_sq / denominator
        
        com_sequence = [initial_com.copy()]
        com_prev = initial_com.copy()
        com_curr = initial_com.copy()
        
        # Calculate initial previous value
        com_prev_step = initial_com - initial_com_vel * dt
        com_sequence.append(com_prev_step)
        com_prev = com_prev_step
        com_curr = initial_com.copy()
        
        for k in range(len(zmp_sequence)):
            if k < len(zmp_sequence):
                zmp_k = zmp_sequence[k]
                
                # Calculate next CoM position
                com_next = a * com_curr + b * com_prev + g * zmp_k
                
                # Update sequence
                com_next_3d = np.array([com_next[0], com_next[1], initial_com[2]])  # Maintain constant height
                com_sequence.append(com_next_3d)
                
                # Update for next iteration
                com_prev = com_curr.copy()
                com_curr = com_next_3d.copy()
        
        return np.array(com_sequence)[2:]  # Remove the first two initial values
```

#### 3. Enhanced Linear Inverted Pendulum Model (ELIPM)

The Enhanced Linear Inverted Pendulum Model accounts for variable CoM height:

```python
class EnhancedLIPMController:
    def __init__(self, nominal_com_height, gravity=9.81):
        self.nominal_com_height = nominal_com_height
        self.gravity = gravity
        self.max_height_variation = 0.1  # Maximum CoM height change
        
        # Control parameters
        self.height_control_p_gain = 5.0
        self.height_control_d_gain = 2.0
        self.com_xy_control_p_gain = 10.0
        self.com_xy_control_d_gain = 4.0
    
    def compute_enhanced_dynamics(self, com_state, zmp_ref, dt):
        """
        Compute CoM acceleration using ELIPM with variable height
        """
        com_pos, com_vel = com_state
        com_x, com_y, com_z = com_pos
        com_vx, com_vy, com_vz = com_vel
        
        # Calculate current pendulum frequency based on actual height
        current_omega = np.sqrt(self.gravity / com_z) if com_z > 0.1 else np.sqrt(self.gravity / self.nominal_com_height)
        
        # XY-plane control (similar to LIPM but with current height)
        xy_error = np.array([com_x, com_y]) - zmp_ref
        xy_acc_open_loop = current_omega**2 * xy_error
        
        # Height control to maintain desired CoM height
        height_error = self.nominal_com_height - com_z
        height_restoring_acc = self.height_control_p_gain * height_error - self.height_control_d_gain * com_vz
        
        # Calculate total CoM acceleration
        com_acc = np.array([
            xy_acc_open_loop[0],  # X acceleration
            xy_acc_open_loop[1],  # Y acceleration
            height_restoring_acc  # Z acceleration
        ])
        
        # Integrate to get new state
        new_com_vel = com_vel + com_acc * dt
        new_com_pos = com_pos + new_com_vel * dt + 0.5 * com_acc * dt**2
        
        return new_com_pos, new_com_vel, com_acc
```

## Zero moment point control

Zero Moment Point (ZMP) control is fundamental to stable bipedal locomotion. The ZMP is the point on the ground where the moment of the ground reaction force is zero. For stable walking, the ZMP must remain within the support polygon defined by the feet.

### ZMP Definition and Calculation

The ZMP is defined as:
$$ x_{ZMP} = x_{CoM} - \frac{h}{g}\ddot{x}_{CoM} $$
$$ y_{ZMP} = y_{CoM} - \frac{h}{g}\ddot{y}_{CoM} $$

Where $ h $ is the CoM height above the ground.

```python
class ZMPController:
    def __init__(self, robot_model, com_height=0.8, gravity=9.81):
        self.robot_model = robot_model
        self.com_height = com_height
        self.gravity = gravity
        
        # ZMP support polygon (changes with foot positions)
        self.support_polygon = []
        self.current_support_foot = 'both'
        
        # Control parameters
        self.zmp_tracking_p_gain = 100.0
        self.zmp_tracking_d_gain = 20.0
        self.com_admittance = 0.01  # How much CoM moves per ZMP error
        
        # For preview control
        self.preview_horizon = 20  # steps
        self.dt = 0.01  # 100Hz control rate
    
    def calculate_zmp(self, wrench):
        """
        Calculate ZMP from ground reaction wrench
        wrench: [fx, fy, fz, mx, my, mz] - force and moment at contact point
        """
        fx, fy, fz, mx, my, mz = wrench
        
        if abs(fz) < 1.0:  # Very small normal force
            # Use CoM position as proxy
            return self.robot_model.get_com_position()[:2]
        
        # ZMP calculation from moments
        zmp_x = -my / fz
        zmp_y = mx / fz
        
        return np.array([zmp_x, zmp_y])
    
    def calculate_zmp_from_com(self, com_pos, com_acc):
        """
        Calculate ZMP from CoM position and acceleration
        """
        h = com_pos[2]  # Current CoM height
        if h <= 0.01:  # Avoid division by zero
            h = self.com_height
        
        zmp_x = com_pos[0] - (h / self.gravity) * com_acc[0]
        zmp_y = com_pos[1] - (h / self.gravity) * com_acc[1]
        
        return np.array([zmp_x, zmp_y])
    
    def calculate_support_polygon(self, left_foot_pos, right_foot_pos):
        """
        Calculate support polygon based on foot positions
        """
        if left_foot_pos is not None and right_foot_pos is not None:
            # Both feet on ground - create polygon encompassing both feet
            # Approximate each foot as a rectangle
            left_points = self.foot_rectangle(left_foot_pos)
            right_points = self.foot_rectangle(right_foot_pos)
            
            all_points = left_points + right_points
            
            # Calculate convex hull (simplified implementation)
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(all_points)
                polygon = [all_points[i] for i in hull.vertices]
            except:
                # If hull computation fails, use simple rectangle
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                polygon = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ]
        elif left_foot_pos is not None:
            # Left foot only support
            polygon = self.foot_rectangle(left_foot_pos)
        elif right_foot_pos is not None:
            # Right foot only support
            polygon = self.foot_rectangle(right_foot_pos)
        else:
            # No support - empty polygon
            polygon = []
        
        return polygon
    
    def foot_rectangle(self, foot_pos):
        """
        Create rectangular approximation of foot
        """
        x, y, z = foot_pos
        # Standard foot dimensions (0.25m x 0.15m)
        half_length = 0.125
        half_width = 0.075
        
        return [
            [x - half_length, y - half_width],
            [x + half_length, y - half_width],
            [x + half_length, y + half_width],
            [x - half_length, y + half_width]
        ]
    
    def is_zmp_stable(self, zmp_pos, support_polygon):
        """
        Check if ZMP is within support polygon
        """
        if len(support_polygon) < 3:
            return False
        
        # Use ray casting algorithm to check if point is inside polygon
        x, y = zmp_pos
        n = len(support_polygon)
        inside = False
        
        p1x, p1y = support_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = support_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def compute_zmp_control(self, desired_zmp, current_zmp, current_com, current_com_vel, dt):
        """
        Compute CoM adjustment to achieve desired ZMP
        """
        # Calculate ZMP error
        zmp_error = desired_zmp - current_zmp
        
        # Use admittance control to determine CoM adjustment
        # CoM should move in direction to correct ZMP error
        com_adjustment = self.com_admittance * zmp_error
        
        # Calculate required CoM acceleration to achieve desired ZMP
        h = current_com[2]
        com_acc_x = self.gravity / h * (current_com[0] - desired_zmp[0])
        com_acc_y = self.gravity / h * (current_com[1] - desired_zmp[1])
        com_acc_z = 0  # Typically keep CoM height constant
        
        desired_com_acc = np.array([com_acc_x, com_acc_y, com_acc_z])
        
        # Use feedback to refine the acceleration command
        # This implements a form of ZMP feedback control
        feedback_term = self.zmp_tracking_p_gain * zmp_error + self.zmp_tracking_d_gain * (-current_com_vel[:2])
        
        # Calculate final CoM acceleration command
        final_com_acc = desired_com_acc[:2] + feedback_term
        final_com_acc = np.append(final_com_acc, [0])  # Add Z component (0)
        
        # Integrate to get velocity and position commands
        new_com_vel = current_com_vel + final_com_acc * dt
        new_com_pos = current_com + new_com_vel * dt + 0.5 * final_com_acc * dt**2
        
        return new_com_pos, new_com_vel, final_com_acc
```

### Preview Control for ZMP

Preview control uses future ZMP references to generate smoother CoM trajectories:

```python
class ZMPPreviewController:
    def __init__(self, com_height, step_time=0.8, gravity=9.81, dt=0.01):
        self.com_height = com_height
        self.step_time = step_time
        self.gravity = gravity
        self.dt = dt
        self.omega = np.sqrt(gravity / com_height)
        
        # Calculate number of steps in preview horizon
        self.preview_steps = int(2.0 / dt)  # 2 second preview
        
        # Pre-compute preview control gains
        self._compute_preview_gains()
    
    def _compute_preview_gains(self):
        """
        Compute preview control gains based on system dynamics
        """
        # The preview control law is:
        # u(k) = Kx(k) + \sum_{j=0}^{N-1} F(j)r(k+j)
        # where u is control input, x is state, r is reference, K and F are gains
        
        # For LIPM, the continuous-time solution gives us the discrete-time gains
        dt = self.dt
        omega = self.omega
        
        # State-space representation of LIPM:
        # dx/dt = Ax + Bu
        # y = Cx
        # where x = [r_com, dr_com], u = zmp_ref, y = zmp_realized
        
        A = np.array([
            [0, 1, 0, 0],
            [omega**2, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, omega**2, 0]
        ])
        
        B = np.array([
            [0],
            [-omega**2],
            [0],
            [0]
        ])
        
        # Discretize the system
        I = np.eye(4)
        Phi = I + A*dt + (A*dt)@A*dt/2 + (A*dt)@(A*dt)@(A*dt)/6  # Matrix exponential approximation
        Gamma = B*dt  # For small dt
        
        # For preview control, we need to compute the steady-state solution
        # This is a simplified approach - full implementation would require
        # more sophisticated control design
        
        # For now, we'll use a simplified approach based on analytical solution
        self.preview_weights = []
        for j in range(self.preview_steps):
            # Exponential decay weights based on time to go
            t = j * dt
            weight = np.exp(-omega * t)  # Exponentially decreasing influence
            self.preview_weights.append(weight)
    
    def compute_preview_control(self, current_com, current_com_vel, zmp_reference_sequence):
        """
        Compute CoM trajectory using preview control
        """
        # This implements the preview control law for ZMP tracking
        # The idea is to use future ZMP references to generate optimal CoM trajectory
        
        if len(zmp_reference_sequence) < self.preview_steps:
            # Extend the sequence with the last value
            extension = [zmp_reference_sequence[-1]] * (self.preview_steps - len(zmp_reference_sequence))
            zmp_ref_extended = np.vstack([zmp_reference_sequence, extension])
        else:
            zmp_ref_extended = zmp_reference_sequence[:self.preview_steps]
        
        # Calculate the infinite horizon preview control solution
        # This is based on the analytical solution of the LQR problem with preview
        
        # For the LIPM, the optimal solution can be expressed as:
        # r_com[k] = \sum_{j=0}^{k-1} G1(j)e[k-j] + \sum_{j=0}^{N-1} G2(j)r[k+j]
        # where e is tracking error and r is reference
        
        # Simplified implementation using the analytical solution
        omega = self.omega
        dt = self.dt
        
        # Calculate coefficients for the analytical solution
        a1 = np.cosh(omega * self.step_time / 2)
        a2 = np.sinh(omega * self.step_time / 2) / omega
        
        # Initialize trajectory
        com_trajectory = [current_com.copy()]
        com_vel_trajectory = [current_com_vel.copy()]
        
        # Generate trajectory using preview information
        for k in range(len(zmp_ref_extended) - 1):
            # Calculate desired CoM position based on current and future ZMP references
            weighted_sum = np.zeros(2)
            total_weight = 0
            
            for j in range(min(len(zmp_ref_extended) - k, len(self.preview_weights))):
                weight = self.preview_weights[j]
                weighted_sum += weight * zmp_ref_extended[k + j]
                total_weight += weight
            
            if total_weight > 0:
                predicted_zmp = weighted_sum / total_weight
            else:
                predicted_zmp = zmp_ref_extended[k]

            # Calculate CoM position that would realize this ZMP
            current_com_xy = com_trajectory[-1][:2]
            current_com_vel_xy = com_vel_trajectory[-1][:2]

            # Use LIPM relationship with preview adjustment
            dt_step = self.step_time / len(zmp_ref_extended)  # Approximate step

            # Calculate next CoM state using LIPM dynamics with preview
            # This is a simplified implementation - full preview control is more complex
            com_acc_x = self.gravity / self.com_height * (current_com_xy[0] - predicted_zmp[0])
            com_acc_y = self.gravity / self.com_height * (current_com_xy[1] - predicted_zmp[1])

            new_com_vel_xy = current_com_vel_xy + np.array([com_acc_x, com_acc_y]) * dt
            new_com_xy = current_com_xy + new_com_vel_xy * dt + 0.5 * np.array([com_acc_x, com_acc_y]) * dt**2

            # Create full 3D vectors
            new_com = np.array([new_com_xy[0], new_com_xy[1], current_com[2]])
            new_com_vel = np.array([new_com_vel_xy[0], new_com_vel_xy[1], current_com_vel[2]])

            com_trajectory.append(new_com)
            com_vel_trajectory.append(new_com_vel)

        return np.array(com_trajectory), np.array(com_vel_trajectory)
```

## Compliance control

Compliance control allows the robot to adapt its behavior to environmental constraints and maintain safe interaction with the environment. This is particularly important for humanoid robots that need to handle unexpected contacts and disturbances.

### Variable Impedance Control

Variable impedance control adjusts the robot's mechanical impedance (stiffness, damping, inertia) based on task requirements:

```python
class VariableImpedanceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        
        # Default impedance parameters
        self.default_stiffness = {
            'arm': np.diag([1000, 1000, 1000, 100, 100, 50]),  # High stiffness for position control
            'leg': np.diag([2000, 2000, 2000, 200, 200, 100]),  # Higher for stability
            'trunk': np.diag([1500, 1500, 1500, 150, 150, 80])  # Medium for balance
        }
        
        self.default_damping = {
            'arm': np.diag([200, 200, 200, 20, 20, 10]),
            'leg': np.diag([400, 400, 400, 40, 40, 20]),
            'trunk': np.diag([300, 300, 300, 30, 30, 16])
        }
        
        # Task-based impedance modulation
        self.impedance_modulation_rules = {
            'compliance_required': {
                'approach_object': {'multiplier': 0.3, 'joints': 'arm'},
                'physical_interaction': {'multiplier': 0.1, 'joints': 'arm'},
                'walking_balance': {'multiplier': 0.7, 'joints': 'leg'},
                'disturbance_recovery': {'multiplier': 0.5, 'joints': 'trunk'}
            },
            'stiffness_required': {
                'precision_task': {'multiplier': 2.0, 'joints': 'arm'},
                'posture_maintenance': {'multiplier': 1.5, 'joints': 'trunk'},
                'heavy_load': {'multiplier': 1.8, 'joints': 'arm'}
            }
        }
    
    def modulate_impedance(self, current_task, external_wrenches=None, contact_state=None):
        """
        Modulate impedance based on current task and environmental conditions
        """
        # Start with default impedance
        task_stiffness = {}
        task_damping = {}
        
        for part in ['arm', 'leg', 'trunk']:
            task_stiffness[part] = self.default_stiffness[part].copy()
            task_damping[part] = self.default_damping[part].copy()
        
        # Apply task-based modulation
        if current_task in self.impedance_modulation_rules['compliance_required']:
            rule = self.impedance_modulation_rules['compliance_required'][current_task]
            multiplier = rule['multiplier']
            joints = rule['joints']
            
            if joints == 'all':
                for part in task_stiffness:
                    task_stiffness[part] *= multiplier
                    task_damping[part] *= multiplier
            else:
                task_stiffness[joints] *= multiplier
                task_damping[joints] *= multiplier
        
        elif current_task in self.impedance_modulation_rules['stiffness_required']:
            rule = self.impedance_modulation_rules['stiffness_required'][current_task]
            multiplier = rule['multiplier']
            joints = rule['joints']
            
            if joints == 'all':
                for part in task_stiffness:
                    task_stiffness[part] *= multiplier
                    task_damping[part] *= min(multiplier, 3.0)  # Cap damping increase
            else:
                task_stiffness[joints] *= multiplier
                task_damping[joints] *= min(multiplier, 3.0)
        
        # Apply contact-based modulation if contacts are detected
        if contact_state:
            task_stiffness, task_damping = self.modulate_for_contacts(
                task_stiffness, task_damping, contact_state
            )
        
        # Apply external wrench-based modulation
        if external_wrenches:
            task_stiffness, task_damping = self.modulate_for_external_forces(
                task_stiffness, task_damping, external_wrenches
            )
        
        return task_stiffness, task_damping
    
    def modulate_for_contacts(self, stiffness, damping, contact_state):
        """
        Adjust impedance based on detected contacts
        """
        for link_name, contact_info in contact_state.items():
            if contact_info['contact_detected']:
                contact_force = contact_info['force_magnitude']
                
                # Increase stiffness proportionally to contact force to maintain stability
                force_ratio = min(contact_force / 100.0, 2.0)  # Cap at 2x
                
                # Determine which impedance map to adjust based on link
                if 'arm' in link_name:
                    part = 'arm'
                elif 'leg' in link_name or 'foot' in link_name:
                    part = 'leg'
                else:
                    part = 'trunk'
                
                stiffness[part] *= (1.0 + 0.5 * force_ratio)
                damping[part] *= (1.0 + 0.3 * force_ratio)
        
        return stiffness, damping
    
    def modulate_for_external_forces(self, stiffness, damping, external_wrenches):
        """
        Adjust impedance based on external wrenches applied to the robot
        """
        for link_name, wrench in external_wrenches.items():
            force_magnitude = np.linalg.norm(wrench[:3])
            
            # Adjust impedance based on external force magnitude
            if force_magnitude > 50.0:  # Significant external force
                force_ratio = min(force_magnitude / 100.0, 3.0)  # Cap at 3x
                
                # Determine which impedance map to adjust
                if 'arm' in link_name:
                    part = 'arm'
                elif 'leg' in link_name or 'foot' in link_name:
                    part = 'leg'
                else:
                    part = 'trunk'
                
                # Increase both stiffness and damping to resist external forces
                stiffness[part] *= (1.0 + 0.7 * force_ratio)
                damping[part] *= (1.0 + 0.5 * force_ratio)
        
        return stiffness, damping

class ComplianceBasedBalancer:
    def __init__(self, robot_model, com_height=0.8, gravity=9.81):
        self.robot_model = robot_model
        self.com_height = com_height
        self.gravity = gravity
        
        # Compliance controller
        self.compliance_controller = VariableImpedanceController(robot_model)
        
        # Balance-specific parameters
        self.compliance_bandwidth = 5.0  # Hz
        self.disturbance_observer_cutoff = 2.0  # Hz
        self.contact_threshold = 5.0  # N for contact detection
        
        # Internal state
        self.last_com_position = np.zeros(3)
        self.last_com_velocity = np.zeros(3)
        self.disturbance_estimate = np.zeros(3)
        self.contact_forces = {}
    
    def compute_compliant_balance_control(self, desired_com, current_state, task_context):
        """
        Compute compliant balance control commands
        """
        # Extract current state
        current_com = current_state['com_position']
        current_com_vel = current_state['com_velocity']
        current_joint_positions = current_state['joint_positions']
        current_joint_velocities = current_state['joint_velocities']
        external_wrenches = current_state.get('external_wrenches', {})
        contact_sensors = current_state.get('contact_sensors', {})
        
        # Estimate disturbances affecting balance
        self.update_disturbance_estimate(current_com, current_com_vel)
        
        # Determine appropriate impedance based on task and environment
        task_stiffness, task_damping = self.compliance_controller.modulate_impedance(
            task_context['current_task'],
            external_wrenches,
            contact_sensors
        )
        
        # Calculate balance control with compliance
        balance_commands = self.compute_balance_with_compliance(
            desired_com, current_com, current_com_vel,
            task_stiffness, task_damping, self.disturbance_estimate
        )
        
        # Apply joint-level compliance control
        joint_commands = self.compute_joint_compliance_control(
            balance_commands, current_joint_positions, current_joint_velocities,
            task_stiffness, task_damping
        )
        
        return joint_commands
    
    def update_disturbance_estimate(self, current_com, current_com_vel):
        """
        Estimate external disturbances affecting the robot's balance
        """
        # Simple first-order disturbance observer
        # In practice, this would use more sophisticated filtering
        
        # Calculate expected CoM motion based on previous state
        dt = 0.01  # Assume 100Hz control rate
        expected_com_vel = self.last_com_velocity
        expected_com_pos = self.last_com_position + expected_com_vel * dt
        
        # Calculate disturbance based on deviation from expected motion
        position_error = current_com - expected_com_pos
        velocity_error = current_com_vel - expected_com_vel
        
        # Update disturbance estimate with low-pass filtering
        alpha = 0.1  # Low-pass filter coefficient
        self.disturbance_estimate = (1 - alpha) * self.disturbance_estimate + \
                                   alpha * (position_error / dt + velocity_error)
        
        # Update internal state
        self.last_com_position = current_com.copy()
        self.last_com_velocity = current_com_vel.copy()
    
    def compute_balance_with_compliance(self, desired_com, current_com, current_com_vel,
                                      stiffness_map, damping_map, disturbance_estimate):
        """
        Compute balance control commands with compliance
        """
        # Calculate position and velocity errors
        pos_error = desired_com - current_com
        vel_error = -current_com_vel  # Drive velocity to zero
        
        # Calculate compliant balance forces
        # These will be translated to joint torques later
        stiffness_matrix = self.get_stiffness_for_balance(stiffness_map)
        damping_matrix = self.get_damping_for_balance(damping_map)
        
        # Apply stiffness and damping to errors
        position_feedback = stiffness_matrix @ pos_error
        velocity_feedback = damping_matrix @ vel_error
        
        # Include disturbance compensation
        disturbance_compensation = -disturbance_estimate * 0.5  # Partial compensation
        
        # Total balance command in Cartesian space
        balance_command = position_feedback + velocity_feedback + disturbance_compensation
        
        return {
            'cartesian_command': balance_command,
            'position_error': pos_error,
            'velocity_error': vel_error,
            'disturbance_compensation': disturbance_compensation
        }
    
    def get_stiffness_for_balance(self, stiffness_map):
        """
        Extract appropriate stiffness matrix for balance control
        """
        # For balance, we primarily care about X and Y directions
        # and to some extent Z for height control
        stiffness = np.zeros((6, 6))  # 6 DOF: 3 translational, 3 rotational
        
        # Position control stiffness (XY mainly for balance)
        stiffness[0, 0] = stiffness_map['trunk'][0, 0] * 0.8  # X (forward/back)
        stiffness[1, 1] = stiffness_map['trunk'][1, 1] * 0.8  # Y (lateral)
        stiffness[2, 2] = stiffness_map['trunk'][2, 2] * 0.3  # Z (height - less aggressive)
        
        # Orientation control stiffness (for trunk stability)
        stiffness[3, 3] = stiffness_map['trunk'][3, 3] * 0.5  # Roll
        stiffness[4, 4] = stiffness_map['trunk'][4, 4] * 0.5  # Pitch
        stiffness[5, 5] = stiffness_map['trunk'][5, 5] * 0.2  # Yaw (least constrained)
        
        return stiffness
    
    def get_damping_for_balance(self, damping_map):
        """
        Extract appropriate damping matrix for balance control
        """
        damping = np.zeros((6, 6))
        
        # Use similar structure as stiffness but with appropriate damping values
        damping[0, 0] = damping_map['trunk'][0, 0] * 0.8
        damping[1, 1] = damping_map['trunk'][1, 1] * 0.8
        damping[2, 2] = damping_map['trunk'][2, 2] * 0.3
        damping[3, 3] = damping_map['trunk'][3, 3] * 0.5
        damping[4, 4] = damping_map['trunk'][4, 4] * 0.5
        damping[5, 5] = damping_map['trunk'][5, 5] * 0.2
        
        return damping
    
    def compute_joint_compliance_control(self, balance_commands, joint_positions, 
                                       joint_velocities, stiffness_map, damping_map):
        """
        Translate Cartesian balance commands to joint-space compliance control
        """
        # Calculate Jacobian for the trunk/base link
        trunk_jacobian = self.robot_model.get_link_jacobian('trunk', joint_positions)
        
        # Map Cartesian commands to joint space
        # tau = J^T * F where F is the Cartesian force command
        cartesian_force = balance_commands['cartesian_command']
        joint_torques = trunk_jacobian.T @ cartesian_force
        
        # Add joint-level compliance control
        joint_commands = {}
        
        # Apply variable impedance to each joint based on its role
        for joint_name, joint_idx in self.robot_model.joint_indices.items():
            if joint_name in joint_positions:
                # Determine which impedance map to use based on joint location
                if 'arm' in joint_name:
                    part = 'arm'
                elif 'leg' in joint_name or 'hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name:
                    part = 'leg'
                else:
                    part = 'trunk'
                
                # Get joint-specific stiffness and damping
                # Use diagonal elements corresponding to this joint
                stiffness = stiffness_map[part][joint_idx, joint_idx] if joint_idx < stiffness_map[part].shape[0] else 1000
                damping = damping_map[part][joint_idx, joint_idx] if joint_idx < damping_map[part].shape[0] else 100
                
                # Calculate compliance control command for this joint
                desired_position = joint_positions[joint_name]  # Or some desired position
                position_error = desired_position - joint_positions[joint_name]
                velocity_error = -joint_velocities[joint_name]  # Drive to zero velocity
                
                compliance_torque = (stiffness * position_error + 
                                   damping * velocity_error)
                
                # Add to the joint torque from Cartesian control
                total_torque = joint_torques[joint_idx] if joint_idx < len(joint_torques) else 0
                total_torque += compliance_torque
                
                joint_commands[joint_name] = total_torque
        
        return joint_commands
```

### Admittance Control

Admittance control allows the robot to behave like a spring-mass-damper system in response to external forces:

```python
class AdmittanceController:
    def __init__(self, robot_model, admittance_params=None):
        self.robot_model = robot_model
        
        # Default admittance parameters (M, B, K - mass, damping, stiffness)
        if admittance_params is None:
            self.admittance_params = {
                'mass_matrix': np.diag([10.0, 10.0, 8.0, 0.1, 0.1, 0.05]),  # Lower mass for more responsive
                'damping_matrix': np.diag([200.0, 200.0, 150.0, 20.0, 20.0, 10.0]),  # Higher damping for stability
                'stiffness_matrix': np.diag([1000.0, 1000.0, 800.0, 100.0, 100.0, 50.0])  # Stiffness for position
            }
        else:
            self.admittance_params = admittance_params
        
        # Internal state for velocity and position integration
        self.last_external_wrench = np.zeros(6)
        self.integrated_velocity = np.zeros(6)
        self.integrated_position = np.zeros(6)
        
        # Filter parameters for force measurement
        self.force_lowpass_coeff = 0.1
        self.filtered_external_wrench = np.zeros(6)
    
    def update_admittance_control(self, external_wrench, dt=0.01):
        """
        Update admittance control based on external wrench
        external_wrench: [fx, fy, fz, mx, my, mz] in end-effector frame
        """
        # Low-pass filter the external wrench to reduce noise
        self.filtered_external_wrench = (
            self.force_lowpass_coeff * external_wrench +
            (1 - self.force_lowpass_coeff) * self.filtered_external_wrench
        )
        
        # Calculate admittance response: v_dot = M^(-1) * (F_ext - B*v - K*x)
        # where v is velocity, x is position, F_ext is external force
        
        # Calculate the force term in the admittance equation
        damping_force = self.admittance_params['damping_matrix'] @ self.integrated_velocity
        stiffness_force = self.admittance_params['stiffness_matrix'] @ self.integrated_position
        
        net_force = (self.filtered_external_wrench - 
                    damping_force - 
                    stiffness_force)
        
        # Calculate acceleration: a = M^(-1) * net_force
        mass_inv = np.linalg.inv(self.admittance_params['mass_matrix'])
        acceleration = mass_inv @ net_force
        
        # Integrate to get velocity and position
        new_velocity = self.integrated_velocity + acceleration * dt
        new_position = self.integrated_position + new_velocity * dt
        
        # Update internal state
        self.integrated_velocity = new_velocity
        self.integrated_position = new_position
        
        # Limit the integrated position to prevent drift
        position_limit = 0.1  # 10 cm maximum displacement
        self.integrated_position = np.clip(self.integrated_position, 
                                         -position_limit, position_limit)
        
        return {
            'desired_velocity': new_velocity,
            'desired_position': new_position,
            'applied_force_compensation': net_force
        }
    
    def adapt_admittance_parameters(self, task_context, environment_state):
        """
        Adapt admittance parameters based on task and environment
        """
        new_params = {
            'mass_matrix': self.admittance_params['mass_matrix'].copy(),
            'damping_matrix': self.admittance_params['damping_matrix'].copy(),
            'stiffness_matrix': self.admittance_params['stiffness_matrix'].copy()
        }
        
        # Adjust parameters based on task requirements
        if task_context.get('task_type') == 'delicate_manipulation':
            # More compliant for delicate tasks
            new_params['mass_matrix'] *= 0.5  # More responsive
            new_params['damping_matrix'] *= 0.7  # Less damping
            new_params['stiffness_matrix'] *= 0.3  # More compliant
        elif task_context.get('task_type') == 'stiff_positioning':
            # Stiffer for precise positioning
            new_params['mass_matrix'] *= 1.5  # Less responsive (more stable)
            new_params['damping_matrix'] *= 1.2  # More damping
            new_params['stiffness_matrix'] *= 2.0  # Stiffer
        elif task_context.get('task_type') == 'balance_recovery':
            # Adaptive based on balance state
            if environment_state.get('is_balanced', True):
                # Normal admittance
                pass
            else:
                # More compliant to absorb disturbances
                new_params['stiffness_matrix'] *= 0.5
                new_params['damping_matrix'] *= 0.8
        
        # Adjust based on contact state
        contact_state = environment_state.get('contact_state', {})
        if contact_state.get('is_in_contact', False):
            if contact_state.get('contact_force', 0) > 100:  # High contact force
                # Reduce stiffness to prevent damage
                new_params['stiffness_matrix'] *= 0.7
                new_params['damping_matrix'] *= 0.9
        
        return new_params

# Example usage of admittance control in balance context
class CompliantBalanceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.admittance_controller = AdmittanceController(robot_model)
        self.compliance_balancer = ComplianceBasedBalancer(robot_model)
        
        # Task context
        self.current_task = 'walking'
        self.environment_state = {
            'is_balanced': True,
            'contact_state': {'is_in_contact': False, 'contact_force': 0},
            'disturbance_level': 0.0
        }
    
    def compute_balance_with_admittance(self, robot_state, desired_com):
        """
        Compute balance control with admittance for external disturbances
        """
        # First, compute standard balance control
        balance_commands = self.compliance_balancer.compute_compliant_balance_control(
            desired_com, robot_state, 
            {'current_task': self.current_task}
        )
        
        # Then apply admittance control to handle external forces
        external_wrenches = robot_state.get('external_wrenches', {})
        
        # Aggregate external forces (simplified - just use trunk force if available)
        if 'trunk' in external_wrenches:
            external_wrench = np.array(external_wrenches['trunk'])
        else:
            external_wrench = np.zeros(6)  # No external forces
        
        # Update admittance controller
        admittance_response = self.admittance_controller.update_admittance_control(
            external_wrench, dt=0.01
        )
        
        # Adapt admittance parameters based on situation
        new_admittance_params = self.admittance_controller.adapt_admittance_parameters(
            {'task_type': self.current_task},
            self.environment_state
        )
        self.admittance_controller.admittance_params = new_admittance_params
        
        # Combine balance commands with admittance response
        # This is a simplified combination - in practice, this would be more sophisticated
        final_commands = balance_commands.copy()
        
        # Add admittance-based adjustments to joint commands
        for joint_name, torque in final_commands.items():
            # Add small adjustment based on admittance response
            admittance_adjustment = admittance_response['applied_force_compensation'][0] * 0.01  # Scale appropriately
            final_commands[joint_name] += admittance_adjustment
        
        return final_commands
```

## Multi-objective optimization

Whole body control often involves multiple competing objectives that must be optimized simultaneously. This requires advanced optimization techniques to balance different control goals.

### Quadratic Programming for Whole Body Control

Quadratic programming (QP) is commonly used for multi-objective whole body control because it can handle multiple objectives and constraints efficiently.

```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class WholeBodyQPSolver:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = len(robot_model.joint_names)
        
        # Default weights for different objectives
        self.weights = {
            'tracking': 1.0,      # Priority for tracking desired motions
            'com_balance': 5.0,   # Priority for CoM balance
            'zmp_tracking': 3.0,  # Priority for ZMP control
            'energy_effort': 0.01,  # Priority for minimizing energy
            'joint_limits': 10.0,   # Penalty for approaching limits
            'smoothness': 0.1,      # Priority for smooth motion
            'contact_stability': 5.0  # Priority for maintaining contacts
        }
    
    def formulate_qp_problem(self, state, objectives, constraints):
        """
        Formulate the whole-body control problem as a QP
        """
        # Decision variables: joint accelerations (and possibly contact forces)
        n_vars = self.n_joints  # For now, just joint accelerations
        
        # Quadratic cost function: J = 0.5 * x^T * P * x + q^T * x
        # where x = [joint_acc, contact_forces, ...]
        
        # Initialize cost matrices
        P = np.zeros((n_vars, n_vars))
        q = np.zeros(n_vars)
        
        # Add each objective to the cost function
        for obj_name, obj_weight in self.weights.items():
            if obj_name in objectives:
                obj_P, obj_q = self.compute_objective_terms(
                    obj_name, state, objectives[obj_name]
                )
                P += obj_weight * obj_P
                q += obj_weight * obj_q
        
        # Constraints: A_eq * x = b_eq, A_ub * x <= b_ub
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        for constraint in constraints:
            if constraint['type'] == 'equality':
                A_eq.append(constraint['A'])
                b_eq.append(constraint['b'])
            elif constraint['type'] == 'inequality':
                A_ub.append(constraint['A'])
                b_ub.append(constraint['b'])
        
        # Convert to numpy arrays
        A_eq = np.array(A_eq) if A_eq else np.zeros((0, n_vars))
        b_eq = np.array(b_eq) if b_eq else np.zeros(0)
        A_ub = np.array(A_ub) if A_ub else np.zeros((0, n_vars))
        b_ub = np.array(b_ub) if b_ub else np.zeros(0)
        
        return P, q, A_eq, b_eq, A_ub, b_ub
    
    def compute_objective_terms(self, obj_name, state, obj_params):
        """
        Compute quadratic and linear terms for a specific objective
        """
        n = self.n_joints
        
        if obj_name == 'tracking':
            # Minimize deviation from desired joint accelerations
            # || ddq - ddq_des ||^2
            P = np.eye(n)
            q = -2 * obj_params['desired_accelerations']  # ddq_des
            
        elif obj_name == 'com_balance':
            # Minimize CoM deviation from desired position
            # Use Jacobian to relate joint accelerations to CoM acceleration
            com_jacobian = self.robot_model.get_com_jacobian(state['joint_positions'])
            
            # CoM acc = J_com * ddq + bias
            # Want CoM_acc to be such that CoM goes to desired position
            P = com_jacobian.T @ com_jacobian
            desired_com_acc = obj_params['desired_com_acceleration']
            q = -2 * com_jacobian.T @ desired_com_acc
            
        elif obj_name == 'zmp_tracking':
            # Minimize ZMP deviation from desired
            # More complex - involves both kinematics and dynamics
            zmp_jacobian = self.compute_zmp_jacobian(state)
            
            P = zmp_jacobian.T @ zmp_jacobian
            desired_zmp_acc = obj_params['desired_zmp_acceleration']
            q = -2 * zmp_jacobian.T @ desired_zmp_acc
            
        elif obj_name == 'energy_effort':
            # Minimize joint effort: || ddq ||^2
            P = 0.01 * np.eye(n)  # Small weight to regularize
            q = np.zeros(n)
            
        elif obj_name == 'joint_limits':
            # Penalize approach to joint limits
            # This is a simplified version - in reality would be more complex
            joint_positions = state['joint_positions']
            P = np.zeros((n, n))
            q = np.zeros(n)
            
            for i, joint_name in enumerate(self.robot_model.joint_names):
                pos = joint_positions[joint_name]
                lower_limit = self.robot_model.joint_limits[joint_name][0]
                upper_limit = self.robot_model.joint_limits[joint_name][1]
                
                # Add penalty for approaching limits
                if pos < lower_limit + 0.1:  # Within 0.1 rad of limit
                    P[i, i] += 1000  # High penalty
                    q[i] += -2 * 1000 * (lower_limit + 0.05)  # Bias toward middle
                elif pos > upper_limit - 0.1:
                    P[i, i] += 1000  # High penalty
                    q[i] += -2 * 1000 * (upper_limit - 0.05)  # Bias toward middle
                    
        elif obj_name == 'contact_stability':
            # Ensure contact forces remain in friction cones
            # Simplified version - would involve contact Jacobians and friction coefficients
            contact_jacobians = obj_params.get('contact_jacobians', [])
            P = np.zeros((n, n))
            q = np.zeros(n)
            
            for jacobian in contact_jacobians:
                # Add regularization to maintain contact stability
                P += 0.1 * jacobian.T @ jacobian
                
        else:
            # Default: minimize acceleration norm
            P = np.eye(n)
            q = np.zeros(n)
        
        return P, q
    
    def compute_zmp_jacobian(self, state):
        """
        Compute Jacobian relating joint accelerations to ZMP accelerations
        """
        # This is a simplified approximation
        # In reality, this would involve complex derivatives of the ZMP equation
        
        joint_positions = state['joint_positions']
        com_position = state['com_position']
        
        # Get CoM Jacobian
        com_jacobian = self.robot_model.get_com_jacobian(joint_positions)
        
        # Approximate ZMP Jacobian from CoM Jacobian
        # ZMP_x = CoM_x - (h/g) * CoM_acc_x
        # So d(ZMP_x)/d(ddq) ≈ d(CoM_x)/d(ddq) - (h/g) * d(CoM_acc_x)/d(ddq)
        # But d(CoM_acc_x)/d(ddq) involves the derivative of the CoM Jacobian
        
        # For simplicity, we'll use a scaled version of the CoM Jacobian
        # In practice, this would be computed more accurately
        zmp_jacobian = 0.8 * com_jacobian  # Scaled to account for the ZMP relationship
        
        return zmp_jacobian
    
    def solve_whole_body_control(self, state, tasks, constraints=None):
        """
        Solve the whole-body control problem using QP
        """
        if constraints is None:
            constraints = []
        
        # Formulate the QP problem
        P, q, A_eq, b_eq, A_ub, b_ub = self.formulate_qp_problem(state, tasks, constraints)
        
        # Solve the QP using cvxpy
        n_vars = P.shape[0]
        x = cp.Variable(n_vars)
        
        # Define the quadratic objective
        objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
        
        # Define constraints
        constraints_qp = []
        if A_eq.shape[0] > 0:
            constraints_qp.append(A_eq @ x == b_eq)
        if A_ub.shape[0] > 0:
            constraints_qp.append(A_ub @ x <= b_ub)
        
        # Solve the problem
        prob = cp.Problem(objective, constraints_qp)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if prob.status not in ["infeasible", "unbounded"]:
                solution = x.value
                return {
                    'joint_accelerations': solution[:self.n_joints],
                    'status': prob.status,
                    'optimal_value': prob.value
                }
            else:
                # Handle infeasible solution
                return {
                    'joint_accelerations': np.zeros(self.n_joints),
                    'status': prob.status,
                    'optimal_value': float('inf')
                }
        except Exception as e:
            print(f"QP solver failed: {e}")
            return {
                'joint_accelerations': np.zeros(self.n_joints),
                'status': 'error',
                'optimal_value': float('inf'),
                'error': str(e)
            }

# Example usage of the QP solver
class MultiObjectiveBalancer:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.qp_solver = WholeBodyQPSolver(robot_model)
        
        # Task priorities and weights
        self.task_weights = {
            'com_tracking': 10.0,
            'zmp_control': 8.0,
            'posture_reference': 2.0,
            'joint_limit_avoidance': 5.0,
            'energy_efficiency': 0.5
        }
    
    def compute_multi_objective_control(self, robot_state, desired_trajectories):
        """
        Compute whole-body control using multi-objective optimization
        """
        # Define control tasks
        tasks = {
            'com_tracking': {
                'desired_position': desired_trajectories['com_position'],
                'desired_velocity': desired_trajectories['com_velocity'],
                'desired_acceleration': desired_trajectories['com_acceleration']
            },
            'zmp_control': {
                'desired_zmp': desired_trajectories['zmp'],
                'current_zmp': self.calculate_current_zmp(robot_state)
            },
            'posture_reference': {
                'reference_posture': desired_trajectories['joint_positions'],
                'stiffness': 100  # How strongly to track reference posture
            },
            'joint_limit_avoidance': {
                'current_positions': robot_state['joint_positions']
            }
        }
        
        # Define constraints
        constraints = self.define_balance_constraints(robot_state)
        
        # Solve the QP problem
        solution = self.qp_solver.solve_whole_body_control(
            robot_state, tasks, constraints
        )
        
        if solution['status'] == 'optimal':
            # Convert joint accelerations to torques using inverse dynamics
            joint_accelerations = solution['joint_accelerations']
            joint_torques = self.compute_feedforward_torques(
                robot_state, joint_accelerations
            )
            
            return {
                'joint_commands': joint_torques,
                'joint_accelerations': joint_accelerations,
                'optimization_status': solution['status'],
                'cost_value': solution['optimal_value']
            }
        else:
            # Handle non-optimal solution
            return {
                'joint_commands': self.compute_safe_posture_torques(robot_state),
                'joint_accelerations': np.zeros(self.qp_solver.n_joints),
                'optimization_status': solution['status'],
                'cost_value': solution['optimal_value'],
                'warning': 'Using safe posture due to optimization failure'
            }
    
    def define_balance_constraints(self, robot_state):
        """
        Define constraints for balance maintenance
        """
        constraints = []
        
        # Joint limit constraints
        for i, joint_name in enumerate(self.robot_model.joint_names):
            lower_limit = self.robot_model.joint_limits[joint_name][0]
            upper_limit = self.robot_model.joint_limits[joint_name][1]
            
            # Constraint: lower_limit <= joint_pos + vel*dt + 0.5*acc*dt^2 <= upper_limit
            # For small dt, this approximates to: lower_limit <= joint_pos <= upper_limit
            # We'll implement this as inequality constraints on accelerations
            
            constraint_lower = {
                'type': 'inequality',
                'A': np.zeros(self.qp_solver.n_joints),
                'b': lower_limit - robot_state['joint_positions'][joint_name] - 
                     robot_state['joint_velocities'][joint_name] * 0.01
            }
            constraint_lower['A'][i] = -0.5 * 0.01**2  # Negative because of <= constraint
            
            constraint_upper = {
                'type': 'inequality',
                'A': np.zeros(self.qp_solver.n_joints),
                'b': upper_limit - robot_state['joint_positions'][joint_name] - 
                     robot_state['joint_velocities'][joint_name] * 0.01
            }
            constraint_upper['A'][i] = 0.5 * 0.01**2  # Positive because of <= constraint
            
            constraints.extend([constraint_lower, constraint_upper])
        
        # Contact stability constraints (if in contact)
        if robot_state.get('left_foot_contact', False) or robot_state.get('right_foot_contact', False):
            # Add constraints to maintain positive normal forces
            # This is simplified - in reality would involve friction cones
            pass
        
        return constraints
    
    def calculate_current_zmp(self, robot_state):
        """
        Calculate current ZMP from robot state
        """
        com_pos = robot_state['com_position']
        com_acc = robot_state['com_acceleration']
        zmp_x = com_pos[0] - (com_pos[2] / 9.81) * com_acc[0]
        zmp_y = com_pos[1] - (com_pos[2] / 9.81) * com_acc[1]
        return np.array([zmp_x, zmp_y])
    
    def compute_feedforward_torques(self, robot_state, joint_accelerations):
        """
        Compute joint torques using inverse dynamics
        """
        # Use robot model to compute inverse dynamics
        # tau = M(q)*ddq + C(q,dq)*dq + g(q)
        
        joint_positions = robot_state['joint_positions']
        joint_velocities = robot_state['joint_velocities']
        
        # Get robot model matrices
        M = self.robot_model.get_mass_matrix(joint_positions)
        C = self.robot_model.get_coriolis_matrix(joint_positions, joint_velocities)
        g = self.robot_model.get_gravity_vector(joint_positions)
        
        # Compute required torques
        tau = M @ joint_accelerations + C @ joint_velocities + g
        
        return tau
    
    def compute_safe_posture_torques(self, robot_state):
        """
        Compute safe posture torques when optimization fails
        """
        # Return torques that maintain a stable posture
        safe_torques = {}
        
        for joint_name in self.robot_model.joint_names:
            # Move toward neutral position with low stiffness
            current_pos = robot_state['joint_positions'][joint_name]
            neutral_pos = self.robot_model.neutral_positions.get(joint_name, 0.0)
            
            # Simple PD control toward neutral position
            position_error = neutral_pos - current_pos
            velocity = robot_state['joint_velocities'].get(joint_name, 0.0)
            
            torque = 50 * position_error - 5 * velocity  # PD controller
            safe_torques[joint_name] = torque
        
        return safe_torques
```

### Task Prioritization and Hierarchical Control

For complex humanoid robots, we often need to handle multiple tasks with different priorities:

```python
class HierarchicalTaskController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.n_joints = len(robot_model.joint_names)
        
        # Define task hierarchy
        self.task_hierarchy = [
            'balance',      # Highest priority
            'collision_avoidance',
            'posture',
            'end_effector_tasks',  # Lowest priority
        ]
    
    def solve_hierarchical_control(self, robot_state, task_list):
        """
        Solve control problem with hierarchical task prioritization
        """
        # Sort tasks by priority
        sorted_tasks = sorted(task_list, key=lambda x: x['priority'])
        
        # Initialize nullspace projector
        I = np.eye(self.n_joints)
        current_nullspace_projector = I.copy()
        
        joint_commands = np.zeros(self.n_joints)
        
        for task in sorted_tasks:
            # Project task into current nullspace
            task_command = self.solve_task_in_nullspace(
                task, robot_state, current_nullspace_projector
            )
            
            # Add task command to total command
            joint_commands += current_nullspace_projector @ task_command
            
            # Update nullspace projector to exclude this task's Jacobian
            task_jacobian = self.get_task_jacobian(task, robot_state)
            if task_jacobian.size > 0:  # If task has associated Jacobian
                # Compute new nullspace projector
                # N_new = N_current * (I - J_task^# * J_task)
                # where J_task^# is the pseudo-inverse of J_task in the current nullspace
                J_effective = task_jacobian @ current_nullspace_projector
                J_pinv = np.linalg.pinv(J_effective)
                task_projector = I - J_pinv @ J_effective
                current_nullspace_projector = current_nullspace_projector @ task_projector
        
        return joint_commands
    
    def solve_task_in_nullspace(self, task, robot_state, nullspace_projector):
        """
        Solve a single task while respecting higher-priority tasks
        """
        if task['type'] == 'balance':
            return self.solve_balance_task(task, robot_state, nullspace_projector)
        elif task['type'] == 'end_effector':
            return self.solve_end_effector_task(task, robot_state, nullspace_projector)
        elif task['type'] == 'posture':
            return self.solve_posture_task(task, robot_state, nullspace_projector)
        elif task['type'] == 'collision_avoidance':
            return self.solve_collision_avoidance_task(task, robot_state, nullspace_projector)
        else:
            # Default: treat as a simple tracking task
            return self.solve_tracking_task(task, robot_state, nullspace_projector)
    
    def solve_balance_task(self, task, robot_state, nullspace_projector):
        """
        Solve balance task with highest priority
        """
        # Use ZMP-based balance control
        current_com = robot_state['com_position']
        current_com_vel = robot_state['com_velocity']
        current_com_acc = robot_state['com_acceleration']
        
        desired_zmp = task['desired_zmp']
        
        # Calculate required CoM acceleration to achieve desired ZMP
        com_height = current_com[2]
        g = 9.81
        
        required_com_acc_x = g / com_height * (current_com[0] - desired_zmp[0])
        required_com_acc_y = g / com_height * (current_com[1] - desired_zmp[1])
        
        required_com_acc = np.array([
            required_com_acc_x,
            required_com_acc_y,
            current_com_acc[2]  # Keep Z acceleration as is
        ])
        
        # Map CoM acceleration to joint accelerations
        com_jacobian = self.robot_model.get_com_jacobian(robot_state['joint_positions'])
        
        # Use nullspace-projected inverse to find joint accelerations
        J_proj = com_jacobian @ nullspace_projector
        J_pinv = np.linalg.pinv(J_proj)
        
        joint_acc = J_pinv @ required_com_acc
        
        return joint_acc
    
    def solve_end_effector_task(self, task, robot_state, nullspace_projector):
        """
        Solve end-effector task (position, orientation, or motion)
        """
        ee_name = task['end_effector']
        desired_property = task['desired_property']  # 'position', 'orientation', 'motion'
        
        # Get current end-effector state
        ee_jacobian = self.robot_model.get_jacobian(ee_name, robot_state['joint_positions'])
        
        if desired_property == 'position':
            current_ee_pos = self.robot_model.get_ee_position(
                ee_name, robot_state['joint_positions']
            )
            position_error = task['desired_position'] - current_ee_pos
            
            # Calculate required end-effector velocity
            desired_ee_vel = position_error * 2.0  # Simple proportional control
            
            # Map to joint velocities using nullspace-projected inverse
            J_proj = ee_jacobian @ nullspace_projector
            J_pinv = np.linalg.pinv(J_proj)
            
            joint_vel = J_pinv @ desired_ee_vel
            
            # Convert to joint accelerations (simplified)
            joint_acc = (joint_vel - robot_state['joint_velocities']) / 0.01  # dt = 0.01s
            
            return joint_acc
        else:
            # Handle other end-effector properties
            return np.zeros(self.n_joints)
    
    def solve_posture_task(self, task, robot_state, nullspace_projector):
        """
        Solve posture task (move toward reference joint configuration)
        """
        reference_posture = task['reference_posture']
        
        # Calculate joint position errors
        joint_errors = []
        for joint_name in self.robot_model.joint_names:
            if joint_name in reference_posture:
                current_pos = robot_state['joint_positions'][joint_name]
                desired_pos = reference_posture[joint_name]
                joint_errors.append(desired_pos - current_pos)
            else:
                joint_errors.append(0.0)
        
        joint_errors = np.array(joint_errors)
        
        # Use nullspace-projected inverse to find joint accelerations
        # For posture tasks, the Jacobian is identity in joint space
        # So we just apply the errors in the nullspace
        posture_gain = 10.0  # How aggressively to track posture
        joint_acc = nullspace_projector @ (posture_gain * joint_errors)
        
        return joint_acc
    
    def solve_collision_avoidance_task(self, task, robot_state, nullspace_projector):
        """
        Solve collision avoidance task
        """
        # This is a simplified implementation
        # In practice, this would involve computing repulsive forces
        # based on distance to obstacles
        
        # For now, return small adjustments away from obstacles
        obstacle_distances = task.get('obstacle_distances', {})
        joint_adjustments = np.zeros(self.n_joints)
        
        for i, joint_name in enumerate(self.robot_model.joint_names):
            if joint_name in obstacle_distances:
                distance = obstacle_distances[joint_name]
                if distance < 0.5:  # Within 50cm of obstacle
                    # Apply repulsive adjustment
                    repulsion_strength = max(0, (0.5 - distance) / 0.5)  # 0 to 1 scaling
                    joint_adjustments[i] = repulsion_strength * 0.1  # Small adjustment
        
        return nullspace_projector @ joint_adjustments
    
    def solve_tracking_task(self, task, robot_state, nullspace_projector):
        """
        Solve general tracking task
        """
        # Generic tracking task solver
        if 'desired_trajectory' in task:
            # Follow a joint or Cartesian trajectory
            current = task['current_value']
            desired = task['desired_value']
            
            error = desired - current
            gain = task.get('gain', 1.0)
            
            # Map error to joint space if needed
            if task.get('space') == 'cartesian':
                jacobian = task.get('jacobian')
                if jacobian is not None:
                    J_pinv = np.linalg.pinv(jacobian @ nullspace_projector)
                    joint_error = J_pinv @ error
                else:
                    joint_error = error[:self.n_joints]  # Assume first n values are joint-related
            else:
                # Joint space tracking
                joint_error = error
            
            return nullspace_projector @ (gain * joint_error)
        else:
            return np.zeros(self.n_joints)

# Example usage of hierarchical controller
hierarchical_controller = HierarchicalTaskController(robot_model)

# Define multiple tasks with different priorities
tasks = [
    {
        'type': 'balance',
        'priority': 1,  # Highest priority
        'desired_zmp': [0.02, 0.0],  # Slightly forward for stability
        'weight': 100.0
    },
    {
        'type': 'collision_avoidance',
        'priority': 2,
        'obstacle_distances': {'left_arm': 0.3, 'right_arm': 0.8},
        'weight': 50.0
    },
    {
        'type': 'posture',
        'priority': 3,
        'reference_posture': {
            'left_hip_pitch': 0.1,
            'right_hip_pitch': 0.1,
            'left_knee_pitch': -0.5,
            'right_knee_pitch': -0.5
        },
        'weight': 10.0
    },
    {
        'type': 'end_effector',
        'priority': 4,  # Lowest priority
        'end_effector': 'left_hand',
        'desired_property': 'position',
        'desired_position': [0.3, 0.2, 0.8],
        'weight': 5.0
    }
]

# Solve hierarchical control
hierarchical_commands = hierarchical_controller.solve_hierarchical_control(
    robot_state, tasks
)

print(f"Hierarchical control commands computed successfully")
print(f"Command vector shape: {hierarchical_commands.shape}")
```

## Conclusion

Whole body control for humanoid robots is a complex but essential aspect of creating stable and capable bipedal systems. By combining center of mass control, zero moment point regulation, compliance mechanisms, and multi-objective optimization, we can create controllers that enable humanoid robots to perform complex tasks while maintaining balance and adapting to environmental constraints.

The key to successful whole body control lies in:
1. Proper modeling of the robot's dynamics and kinematics
2. Effective integration of multiple control objectives
3. Appropriate handling of constraints and priorities
4. Robust estimation and disturbance rejection
5. Adaptive control strategies that respond to changing conditions

As humanoid robotics continues to advance, whole body control techniques will become increasingly sophisticated, enabling robots to perform more complex and human-like behaviors in diverse environments.