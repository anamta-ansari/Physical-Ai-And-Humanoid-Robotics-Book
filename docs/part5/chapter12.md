---
title: Kinematics & Dynamics
sidebar_position: 1
description: Forward kinematics, inverse kinematics, dynamic balance, ZMP theory, and torque control for humanoid robots
---

# Kinematics & Dynamics

## Forward kinematics

Forward kinematics is the process of determining the position and orientation of a robot's end-effector based on the known joint angles. For humanoid robots, this is particularly complex due to the multiple degrees of freedom in each limb and the need to maintain balance across the entire body.

### Mathematical Foundation

The forward kinematics of a robotic manipulator is defined by the relationship between joint space and Cartesian space. For a serial chain of n joints, the transformation from the base to the end-effector is given by:

```
T_base_to_end_effector = T_0_1(q1) × T_1_2(q2) × ... × T_n-1_n(qn)
```

Where `T_i_i+1(qi)` is the homogeneous transformation matrix between adjacent joints.

### Denavit-Hartenberg (DH) Convention

The DH convention provides a systematic method for defining coordinate frames for each joint in a robotic chain:

1. **Link Length (ai)**: Distance along the Xi axis from Zi to Zi+1
2. **Link Offset (di)**: Distance along the Zi axis from Xi-1 to Xi
3. **Link Twist (αi)**: Angle about the Xi axis from Zi-1 to Zi
4. **Joint Angle (θi)**: Angle about the Zi axis from Xi-1 to Xi

For revolute joints: θi is variable
For prismatic joints: di is variable

### Forward Kinematics Implementation

```python
import numpy as np
from math import sin, cos, sqrt, atan2

class HumanoidFK:
    def __init__(self):
        # Humanoid robot parameters (example for a simplified arm)
        self.shoulder_offset = [0.0, 0.15, 0.2]  # Shoulder offset from torso
        self.upper_arm_length = 0.3
        self.forearm_length = 0.25
        self.hand_offset = 0.1
        
        # Joint limits (in radians)
        self.joint_limits = {
            'shoulder_pitch': (-2.0, 2.0),
            'shoulder_roll': (-1.5, 1.5),
            'shoulder_yaw': (-2.0, 2.0),
            'elbow_pitch': (-2.5, 0.5),
            'wrist_pitch': (-1.5, 1.5),
            'wrist_yaw': (-1.5, 1.5)
        }
    
    def dh_transform(self, a, alpha, d, theta):
        """
        Calculate DH transformation matrix
        """
        return np.array([
            [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
            [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics_arm(self, joint_angles):
        """
        Calculate forward kinematics for a 6-DOF arm
        joint_angles: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_pitch, wrist_yaw]
        """
        if len(joint_angles) != 6:
            raise ValueError("Expected 6 joint angles")
        
        # Extract joint angles
        sp, sr, sy, ep, wp, wy = joint_angles
        
        # DH parameters for each joint (a, alpha, d, theta)
        dh_params = [
            (0, -np.pi/2, 0, sp),      # Shoulder pitch
            (0, np.pi/2, 0, sr),       # Shoulder roll
            (self.upper_arm_length, 0, 0, sy),  # Shoulder yaw
            (0, -np.pi/2, 0, ep),      # Elbow pitch
            (self.forearm_length, 0, 0, wp),    # Wrist pitch
            (0, 0, 0, wy)              # Wrist yaw
        ]
        
        # Start with shoulder offset
        T_total = self.translate_matrix(self.shoulder_offset)
        
        # Multiply all transformation matrices
        for a, alpha, d, theta in dh_params:
            T = self.dh_transform(a, alpha, d, theta)
            T_total = np.dot(T_total, T)
        
        # Add hand offset
        T_hand = np.dot(T_total, self.translate_matrix([self.hand_offset, 0, 0]))
        
        return T_total, T_hand
    
    def translate_matrix(self, translation):
        """
        Create a translation matrix
        """
        tx, ty, tz = translation
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
    
    def rotation_matrix_to_euler(self, R):
        """
        Convert rotation matrix to Euler angles (ZYX convention)
        """
        sy = sqrt(R[0,0]**2 + R[1,0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = atan2(R[2,1], R[2,2])
            y = atan2(-R[2,0], sy)
            z = atan2(R[1,0], R[0,0])
        else:
            x = atan2(-R[1,2], R[1,1])
            y = atan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z])

# Example usage
fk_solver = HumanoidFK()
joint_angles = [0.1, 0.2, 0.3, -1.0, 0.1, 0.05]  # Example joint angles
T_elbow, T_hand = fk_solver.forward_kinematics_arm(joint_angles)

print("Hand position:", T_hand[:3, 3])
print("Hand orientation matrix:")
print(T_hand[:3, :3])
```

### Forward Kinematics for Humanoid Legs

For humanoid robots, forward kinematics also applies to leg systems, which is crucial for locomotion and balance:

```python
class HumanoidLegFK:
    def __init__(self, side='left'):
        # Humanoid leg parameters
        self.side = side
        self.hip_offset_x = 0.0
        self.hip_offset_y = 0.15 if side == 'left' else -0.15  # Left vs right leg
        self.hip_offset_z = 0.05  # Offset from hip to first joint
        
        # Leg segment lengths
        self.thigh_length = 0.4
        self.shin_length = 0.4
        self.foot_length = 0.25
        self.foot_height = 0.08
        
        # Joint configuration
        self.joint_names = [
            'hip_yaw_pitch',  # Combined yaw and pitch at hip
            'hip_roll',       # Hip roll
            'hip_pitch',      # Hip pitch
            'knee_pitch',     # Knee pitch
            'ankle_pitch',    # Ankle pitch
            'ankle_roll'      # Ankle roll
        ]
    
    def forward_kinematics_leg(self, joint_angles):
        """
        Calculate forward kinematics for a 6-DOF leg
        joint_angles: [hip_yaw_pitch, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll]
        """
        if len(joint_angles) != 6:
            raise ValueError("Expected 6 joint angles")
        
        hy, hr, hp, kp, ap, ar = joint_angles
        
        # Start from hip position
        T_hip = self.translate_matrix([self.hip_offset_x, self.hip_offset_y, self.hip_offset_z])
        
        # Hip transformations
        T_hip_yaw_pitch = self.rotate_z_matrix(hy)  # Yaw component
        T_hip_yaw_pitch = np.dot(T_hip_yaw_pitch, self.rotate_x_matrix(hp))  # Pitch component
        T_hip_total = np.dot(T_hip, T_hip_yaw_pitch)
        
        # Hip roll
        T_hip_roll = np.dot(T_hip_total, self.rotate_y_matrix(hr))
        
        # Thigh (upper leg)
        T_thigh = np.dot(T_hip_roll, self.translate_matrix([0, 0, -self.thigh_length]))
        
        # Knee pitch
        T_knee = np.dot(T_thigh, self.rotate_x_matrix(kp))
        
        # Shin (lower leg)
        T_shin = np.dot(T_knee, self.translate_matrix([0, 0, -self.shin_length]))
        
        # Ankle pitch
        T_ankle_pitch = np.dot(T_shin, self.rotate_x_matrix(ap))
        
        # Ankle roll
        T_ankle_roll = np.dot(T_ankle_pitch, self.rotate_y_matrix(ar))
        
        # Foot
        T_foot = np.dot(T_ankle_roll, self.translate_matrix([0, 0, -self.foot_height]))
        
        # Foot tip (front of foot)
        T_foot_tip = np.dot(T_foot, self.translate_matrix([self.foot_length/2, 0, 0]))
        
        return {
            'hip': T_hip_total,
            'thigh': T_thigh,
            'knee': T_knee,
            'shin': T_shin,
            'ankle': T_ankle_roll,
            'foot': T_foot,
            'foot_tip': T_foot_tip
        }
    
    def rotate_x_matrix(self, angle):
        """Rotation matrix around X axis"""
        return np.array([
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1]
        ])
    
    def rotate_y_matrix(self, angle):
        """Rotation matrix around Y axis"""
        return np.array([
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1]
        ])
    
    def rotate_z_matrix(self, angle):
        """Rotation matrix around Z axis"""
        return np.array([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    def get_foot_position(self, joint_angles):
        """Get only the foot position for balance calculations"""
        transforms = self.forward_kinematics_leg(joint_angles)
        return transforms['foot_tip'][:3, 3]  # Return only position vector

# Example usage for leg
leg_fk = HumanoidLegFK(side='left')
leg_angles = [0.0, 0.1, -0.3, -0.6, 0.1, 0.05]  # Example leg joint angles
leg_transforms = leg_fk.forward_kinematics_leg(leg_angles)
left_foot_pos = leg_fk.get_foot_position(leg_angles)

print("Left foot position:", left_foot_pos)
```

## Inverse kinematics

Inverse kinematics (IK) is the reverse process of forward kinematics, where we determine the required joint angles to achieve a desired end-effector position and orientation. For humanoid robots, IK is critical for tasks like reaching, stepping, and manipulation.

### Mathematical Approach

The inverse kinematics problem can be formulated as:

```
Given: T_desired (desired end-effector pose)
Find: q = [q1, q2, ..., qn] such that FK(q) ≈ T_desired
```

This is generally solved using:
1. **Analytical methods**: Closed-form solutions for simple kinematic chains
2. **Numerical methods**: Iterative approaches for complex chains
3. **Optimization methods**: Minimizing error with constraints

### Jacobian-Based Inverse Kinematics

The Jacobian matrix relates joint velocities to end-effector velocities:

```
J(q) = ∂f/∂q
```

Where f(q) is the forward kinematics function.

For small displacements:
```
Δx = J(q) * Δq
```

Therefore:
```
Δq = J⁺(q) * Δx
```

Where J⁺ is the pseudoinverse of the Jacobian.

```python
class HumanoidIK:
    def __init__(self):
        self.fk_solver = HumanoidFK()
        self.leg_fk_solver = HumanoidLegFK()
        
        # IK parameters
        self.max_iterations = 100
        self.tolerance = 1e-4
        self.learning_rate = 0.1
        
        # Joint limits
        self.joint_limits = {
            'shoulder_pitch': (-2.0, 2.0),
            'shoulder_roll': (-1.5, 1.5),
            'shoulder_yaw': (-2.0, 2.0),
            'elbow_pitch': (-2.5, 0.5),
            'wrist_pitch': (-1.5, 1.5),
            'wrist_yaw': (-1.5, 1.5)
        }
    
    def calculate_jacobian(self, joint_angles, method='arm'):
        """
        Calculate the geometric Jacobian using numerical differentiation
        """
        n_joints = len(joint_angles)
        J = np.zeros((6, n_joints))  # 6 DoF: 3 position + 3 orientation
        
        # Small perturbation
        eps = 1e-7
        
        # Calculate current end-effector pose
        if method == 'arm':
            _, current_pose = self.fk_solver.forward_kinematics_arm(joint_angles)
        else:
            transforms = self.leg_fk_solver.forward_kinematics_leg(joint_angles)
            current_pose = transforms['foot_tip']
        
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        
        # Calculate Jacobian columns
        for i in range(n_joints):
            # Perturb joint angle
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += eps
            
            # Calculate perturbed pose
            if method == 'arm':
                _, perturbed_pose = self.fk_solver.forward_kinematics_arm(perturbed_angles)
            else:
                transforms = self.leg_fk_solver.forward_kinematics_leg(perturbed_angles)
                perturbed_pose = transforms['foot_tip']
            
            perturbed_pos = perturbed_pose[:3, 3]
            perturbed_rot = perturbed_pose[:3, :3]
            
            # Position contribution
            J[:3, i] = (perturbed_pos - current_pos) / eps
            
            # Orientation contribution (using skew-symmetric matrix)
            # Convert rotation matrices to Euler angles for simplicity
            current_euler = self.fk_solver.rotation_matrix_to_euler(current_rot)
            perturbed_euler = self.fk_solver.rotation_matrix_to_euler(perturbed_rot)
            
            J[3:, i] = (perturbed_euler - current_euler) / eps
        
        return J
    
    def inverse_kinematics_arm(self, target_pose, initial_joints=None, max_iterations=100):
        """
        Solve inverse kinematics for arm using Jacobian transpose method
        """
        if initial_joints is None:
            current_joints = np.zeros(6)  # Default to zero configuration
        else:
            current_joints = np.array(initial_joints)
        
        for iteration in range(max_iterations):
            # Calculate current pose
            _, current_pose = self.fk_solver.forward_kinematics_arm(current_joints)
            
            # Calculate error
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]
            rot_error = self.calculate_rotation_error(target_pose[:3, :3], current_pose[:3, :3])
            
            # Combined error
            error = np.concatenate([pos_error, rot_error])
            
            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                print(f"IK converged after {iteration} iterations")
                break
            
            # Calculate Jacobian
            J = self.calculate_jacobian(current_joints, method='arm')
            
            # Calculate joint adjustments using pseudoinverse
            J_pinv = np.linalg.pinv(J)
            delta_joints = J_pinv @ error * self.learning_rate
            
            # Apply joint limits
            current_joints = self.apply_joint_limits(current_joints + delta_joints)
        
        if iteration == max_iterations - 1:
            print(f"Warning: IK did not converge after {max_iterations} iterations")
            print(f"Final error norm: {np.linalg.norm(error)}")
        
        return current_joints, np.linalg.norm(error)
    
    def inverse_kinematics_leg(self, target_foot_pose, initial_joints=None, max_iterations=100):
        """
        Solve inverse kinematics for leg
        """
        if initial_joints is None:
            current_joints = np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0])  # Default standing pose
        else:
            current_joints = np.array(initial_joints)
        
        for iteration in range(max_iterations):
            # Calculate current foot pose
            transforms = self.leg_fk_solver.forward_kinematics_leg(current_joints)
            current_pose = transforms['foot_tip']
            
            # Calculate error
            pos_error = target_foot_pose[:3, 3] - current_pose[:3, 3]
            rot_error = self.calculate_rotation_error(target_foot_pose[:3, :3], current_pose[:3, :3])
            
            # Combined error
            error = np.concatenate([pos_error, rot_error])
            
            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                print(f"Leg IK converged after {iteration} iterations")
                break
            
            # Calculate Jacobian
            J = self.calculate_jacobian(current_joints, method='leg')
            
            # Calculate joint adjustments
            J_pinv = np.linalg.pinv(J)
            delta_joints = J_pinv @ error * self.learning_rate
            
            # Apply joint limits (example limits)
            joint_limits = [(-0.5, 0.5), (-0.3, 0.3), (-1.5, 0.5), (-2.0, 0.0), (-0.5, 0.5), (-0.3, 0.3)]
            for i, (min_val, max_val) in enumerate(joint_limits):
                current_joints[i] = np.clip(current_joints[i] + delta_joints[i], min_val, max_val)
        
        if iteration == max_iterations - 1:
            print(f"Warning: Leg IK did not converge after {max_iterations} iterations")
        
        return current_joints, np.linalg.norm(error)
    
    def calculate_rotation_error(self, target_rot, current_rot):
        """
        Calculate rotation error using the logarithmic map of rotation matrices
        """
        # Calculate rotation error matrix
        R_error = target_rot @ current_rot.T
        
        # Convert to axis-angle representation
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        
        if abs(angle) < 1e-6:
            # Very small rotation, return zero vector
            return np.zeros(3)
        
        # Calculate axis of rotation
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle
    
    def apply_joint_limits(self, joints):
        """
        Apply joint limits to prevent exceeding physical constraints
        """
        limited_joints = joints.copy()
        
        for i, joint_name in enumerate(self.joint_names):
            if joint_name in self.joint_limits:
                min_val, max_val = self.joint_limits[joint_name]
                limited_joints[i] = np.clip(limited_joints[i], min_val, max_val)
        
        return limited_joints

# Example usage
ik_solver = HumanoidIK()

# Define a target pose for the arm
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.3, 0.2]  # Position: x=0.5, y=0.3, z=0.2
# Add some orientation (rotate around z-axis by 45 degrees)
target_pose[:3, :3] = np.array([
    [0.707, -0.707, 0],
    [0.707, 0.707, 0],
    [0, 0, 1]
])

# Solve inverse kinematics
initial_guess = [0.1, 0.1, 0.1, -1.0, 0.1, 0.1]
solution, error = ik_solver.inverse_kinematics_arm(target_pose, initial_guess)

print("IK Solution:", solution)
print("Final error:", error)

# Verify with forward kinematics
_, fk_pose = ik_solver.fk_solver.forward_kinematics_arm(solution)
print("FK verification - Target:", target_pose[:3, 3])
print("FK verification - Result:", fk_pose[:3, 3])
print("Position error:", np.linalg.norm(target_pose[:3, 3] - fk_pose[:3, 3]))
```

### Optimization-Based Inverse Kinematics

For more complex scenarios with constraints, optimization-based approaches are more appropriate:

```python
from scipy.optimize import minimize
import numpy as np

class OptimizedIK:
    def __init__(self):
        self.fk_solver = HumanoidFK()
        
        # Joint limits
        self.joint_limits = [(-2.0, 2.0), (-1.5, 1.5), (-2.0, 2.0), 
                            (-2.5, 0.5), (-1.5, 1.5), (-1.5, 1.5)]
    
    def ik_objective(self, joints, target_pose):
        """
        Objective function for IK optimization
        """
        # Calculate current pose
        _, current_pose = self.fk_solver.forward_kinematics_arm(joints)
        
        # Position error
        pos_error = np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])
        
        # Orientation error
        rot_error_matrix = target_pose[:3, :3] @ current_pose[:3, :3].T
        # Use Frobenius norm of the rotation error matrix minus identity
        rot_error = np.linalg.norm(rot_error_matrix - np.eye(3), 'fro')
        
        # Total weighted error
        total_error = pos_error + 0.5 * rot_error
        
        return total_error
    
    def solve_ik_optimized(self, target_pose, initial_guess=None):
        """
        Solve IK using optimization with constraints
        """
        if initial_guess is None:
            initial_guess = np.zeros(6)
        
        # Define bounds based on joint limits
        bounds = self.joint_limits
        
        # Optimize
        result = minimize(
            fun=self.ik_objective,
            x0=initial_guess,
            args=(target_pose,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        return result.x, result.fun, result.success

# Example usage of optimization-based IK
opt_ik = OptimizedIK()
opt_solution, opt_error, success = opt_ik.solve_ik_optimized(target_pose)

print("Optimized IK Solution:", opt_solution)
print("Optimized Error:", opt_error)
print("Success:", success)
```

## Dynamic balance

Maintaining dynamic balance is crucial for humanoid robots, especially during locomotion. Dynamic balance involves controlling the robot's center of mass (CoM) and angular momentum to prevent falls while executing movements.

### Center of Mass Calculation

The center of mass of a humanoid robot is calculated as the weighted average of all body segment centers of mass:

```
CoM = Σ(mi * ri) / Σmi
```

Where mi is the mass of segment i and ri is the position vector of segment i's CoM.

```python
class DynamicBalance:
    def __init__(self):
        # Humanoid body segment parameters (mass in kg, CoM offset from joint in meters)
        self.body_segments = {
            'head': {'mass': 5.0, 'com_offset': [0, 0, 0.1]},
            'torso': {'mass': 25.0, 'com_offset': [0, 0, 0.2]},
            'left_upper_arm': {'mass': 2.0, 'com_offset': [0, 0, -0.15]},
            'left_forearm': {'mass': 1.5, 'com_offset': [0, 0, -0.12]},
            'right_upper_arm': {'mass': 2.0, 'com_offset': [0, 0, -0.15]},
            'right_forearm': {'mass': 1.5, 'com_offset': [0, 0, -0.12]},
            'left_thigh': {'mass': 6.0, 'com_offset': [0, 0, -0.2]},
            'left_shin': {'mass': 3.5, 'com_offset': [0, 0, -0.18]},
            'left_foot': {'mass': 1.0, 'com_offset': [0.05, 0, -0.05]},
            'right_thigh': {'mass': 6.0, 'com_offset': [0, 0, -0.2]},
            'right_shin': {'mass': 3.5, 'com_offset': [0, 0, -0.18]},
            'right_foot': {'mass': 1.0, 'com_offset': [0.05, 0, -0.05]}
        }
        
        # FK solvers for limbs
        self.left_arm_fk = HumanoidFK()  # Simplified - would need actual FK for each limb
        self.right_arm_fk = HumanoidFK()
        self.left_leg_fk = HumanoidLegFK(side='left')
        self.right_leg_fk = HumanoidLegFK(side='right')
    
    def calculate_com(self, joint_states, base_pose=np.eye(4)):
        """
        Calculate center of mass of the humanoid robot
        joint_states: Dictionary with joint names and angles
        base_pose: Transformation matrix of the robot's base
        """
        total_mass = 0
        weighted_sum = np.zeros(3)
        
        # Calculate CoM for each body segment
        for segment_name, params in self.body_segments.items():
            segment_com = self.calculate_segment_com(segment_name, joint_states, base_pose)
            
            mass = params['mass']
            total_mass += mass
            weighted_sum += mass * segment_com
        
        # Overall CoM
        com = weighted_sum / total_mass if total_mass > 0 else np.zeros(3)
        
        return com, total_mass
    
    def calculate_segment_com(self, segment_name, joint_states, base_pose):
        """
        Calculate the CoM of a specific body segment
        """
        # This is a simplified implementation
        # In a real system, you'd use forward kinematics to get the exact position
        
        if 'arm' in segment_name:
            # Calculate arm segment CoM based on joint angles
            if 'left' in segment_name:
                arm_joints = self.extract_arm_joints(joint_states, 'left')
                fk_result = self.left_arm_fk.forward_kinematics_arm(arm_joints)
            else:
                arm_joints = self.extract_arm_joints(joint_states, 'right')
                fk_result = self.right_arm_fk.forward_kinematics_arm(arm_joints)
            
            # Get joint position and add CoM offset
            joint_pos = fk_result[1][:3, 3]  # Using hand position as reference
            offset = self.body_segments[segment_name]['com_offset']
            
            # Transform offset to world frame and add to joint position
            # This is simplified - in reality, you'd transform the offset vector properly
            segment_com = joint_pos + np.array(offset)
        
        elif 'leg' in segment_name or 'foot' in segment_name:
            # Calculate leg segment CoM
            if 'left' in segment_name:
                leg_joints = self.extract_leg_joints(joint_states, 'left')
                fk_result = self.left_leg_fk.forward_kinematics_leg(leg_joints)
            else:
                leg_joints = self.extract_leg_joints(joint_states, 'right')
                fk_result = self.right_leg_fk.forward_kinematics_leg(leg_joints)
            
            # Determine which joint to use based on segment
            if 'thigh' in segment_name:
                joint_pos = fk_result['thigh'][:3, 3]
            elif 'shin' in segment_name:
                joint_pos = fk_result['shin'][:3, 3]
            elif 'foot' in segment_name:
                joint_pos = fk_result['foot'][:3, 3]
            else:
                joint_pos = fk_result['hip'][:3, 3]  # default
            
            offset = self.body_segments[segment_name]['com_offset']
            segment_com = joint_pos + np.array(offset)
        
        else:
            # For torso/head, assume fixed offset from base
            base_pos = base_pose[:3, 3]
            offset = self.body_segments[segment_name]['com_offset']
            segment_com = base_pos + np.array(offset)
        
        return segment_com
    
    def extract_arm_joints(self, joint_states, side):
        """
        Extract arm joint angles from full joint state
        """
        prefix = f'{side}_'
        joint_names = [
            f'{prefix}shoulder_pitch',
            f'{prefix}shoulder_roll', 
            f'{prefix}shoulder_yaw',
            f'{prefix}elbow_pitch',
            f'{prefix}wrist_pitch',
            f'{prefix}wrist_yaw'
        ]
        
        angles = []
        for name in joint_names:
            if name in joint_states:
                angles.append(joint_states[name])
            else:
                angles.append(0.0)  # Default angle if not specified
        
        return angles
    
    def extract_leg_joints(self, joint_states, side):
        """
        Extract leg joint angles from full joint state
        """
        prefix = f'{side}_'
        joint_names = [
            f'{prefix}hip_yaw_pitch',
            f'{prefix}hip_roll',
            f'{prefix}hip_pitch',
            f'{prefix}knee_pitch',
            f'{prefix}ankle_pitch',
            f'{prefix}ankle_roll'
        ]
        
        angles = []
        for name in joint_names:
            if name in joint_states:
                angles.append(joint_states[name])
            else:
                angles.append(0.0)  # Default angle if not specified
        
        return angles
    
    def calculate_zero_moment_point(self, com_pos, com_vel, com_acc, z_force, gravity=9.81):
        """
        Calculate Zero Moment Point (ZMP) for dynamic balance
        This is a simplified 2D version (x-y plane)
        """
        # ZMP calculation: ZMP = CoM - (g / (z_force/m)) * (CoM_acc / g)
        # Simplified for 2D: ZMP_x = CoM_x - (CoM_acc_x / (g / height))
        
        # Calculate ZMP position
        zmp_x = com_pos[0] - (com_acc[0] / (gravity / (com_pos[2] - 0.0)))  # Simplified
        zmp_y = com_pos[1] - (com_acc[1] / (gravity / (com_pos[2] - 0.0)))  # Simplified
        
        return np.array([zmp_x, zmp_y, 0.0])
    
    def is_balanced(self, zmp, support_polygon):
        """
        Check if the robot is balanced based on ZMP position
        """
        # Check if ZMP is within support polygon (convex hull of feet contact points)
        return self.point_in_polygon(zmp[:2], support_polygon)
    
    def point_in_polygon(self, point, polygon):
        """
        Check if a 2D point is inside a polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def calculate_support_polygon(self, left_foot_pos, right_foot_pos, foot_size=0.15):
        """
        Calculate support polygon based on foot positions
        """
        # Create a polygon representing the support area of both feet
        if left_foot_pos is not None and right_foot_pos is not None:
            # Both feet on ground - create polygon encompassing both feet
            support_points = [
                [left_foot_pos[0] - foot_size/2, left_foot_pos[1] - foot_size/2],
                [left_foot_pos[0] + foot_size/2, left_foot_pos[1] - foot_size/2],
                [left_foot_pos[0] + foot_size/2, left_foot_pos[1] + foot_size/2],
                [left_foot_pos[0] - foot_size/2, left_foot_pos[1] + foot_size/2],
                [right_foot_pos[0] - foot_size/2, right_foot_pos[1] - foot_size/2],
                [right_foot_pos[0] + foot_size/2, right_foot_pos[1] - foot_size/2],
                [right_foot_pos[0] + foot_size/2, right_foot_pos[1] + foot_size/2],
                [right_foot_pos[0] - foot_size/2, right_foot_pos[1] + foot_size/2]
            ]
            
            # Find convex hull of these points
            from scipy.spatial import ConvexHull
            hull = ConvexHull(support_points)
            return [support_points[i] for i in hull.vertices]
        elif left_foot_pos is not None:
            # Only left foot on ground
            return [
                [left_foot_pos[0] - foot_size/2, left_foot_pos[1] - foot_size/2],
                [left_foot_pos[0] + foot_size/2, left_foot_pos[1] - foot_size/2],
                [left_foot_pos[0] + foot_size/2, left_foot_pos[1] + foot_size/2],
                [left_foot_pos[0] - foot_size/2, left_foot_pos[1] + foot_size/2]
            ]
        elif right_foot_pos is not None:
            # Only right foot on ground
            return [
                [right_foot_pos[0] - foot_size/2, right_foot_pos[1] - foot_size/2],
                [right_foot_pos[0] + foot_size/2, right_foot_pos[1] - foot_size/2],
                [right_foot_pos[0] + foot_size/2, right_foot_pos[1] + foot_size/2],
                [right_foot_pos[0] - foot_size/2, right_foot_pos[1] + foot_size/2]
            ]
        else:
            # No feet on ground - no support
            return []

# Example usage
balance_calculator = DynamicBalance()

# Example joint states (simplified)
joint_states = {
    'left_shoulder_pitch': 0.1, 'left_shoulder_roll': 0.05, 'left_shoulder_yaw': 0.1,
    'left_elbow_pitch': -1.0, 'left_wrist_pitch': 0.05, 'left_wrist_yaw': 0.02,
    'right_shoulder_pitch': 0.1, 'right_shoulder_roll': -0.05, 'right_shoulder_yaw': -0.1,
    'right_elbow_pitch': -1.0, 'right_wrist_pitch': -0.05, 'right_wrist_yaw': -0.02,
    'left_hip_yaw_pitch': 0.0, 'left_hip_roll': 0.05, 'left_hip_pitch': -0.1,
    'left_knee_pitch': 0.5, 'left_ankle_pitch': -0.1, 'left_ankle_roll': 0.02,
    'right_hip_yaw_pitch': 0.0, 'right_hip_roll': -0.05, 'right_hip_pitch': -0.1,
    'right_knee_pitch': 0.5, 'right_ankle_pitch': -0.1, 'right_ankle_roll': -0.02
}

# Calculate CoM
com_pos, total_mass = balance_calculator.calculate_com(joint_states)
print(f"Center of Mass: {com_pos}")
print(f"Total Mass: {total_mass}")

# Calculate support polygon
left_foot_pos = np.array([0.1, 0.15, 0.0])  # Example foot position
right_foot_pos = np.array([0.1, -0.15, 0.0])  # Example foot position
support_polygon = balance_calculator.calculate_support_polygon(left_foot_pos, right_foot_pos)

print(f"Support Polygon: {support_polygon}")

# Calculate ZMP (simplified example)
com_vel = np.array([0.01, 0.0, 0.0])  # Example velocity
com_acc = np.array([0.001, 0.0, 0.0])  # Example acceleration
z_force = total_mass * 9.81  # Example normal force
zmp = balance_calculator.calculate_zero_moment_point(com_pos, com_vel, com_acc, z_force)

print(f"ZMP: {zmp}")

# Check balance
is_balanced = balance_calculator.is_balanced(zmp, support_polygon)
print(f"Is Balanced: {is_balanced}")
```

## ZMP (Zero Moment Point) theory

The Zero Moment Point (ZMP) is a critical concept in humanoid robotics that describes the point on the ground where the moment of the ground reaction force becomes zero. Maintaining the ZMP within the support polygon is essential for dynamic balance.

### ZMP Mathematical Foundation

The ZMP is defined by the following equations:

```
x_ZMP = x_COM - (z_COM - z_support) * ẍ_COM / g
y_ZMP = y_COM - (z_COM - z_support) * ÿ_COM / g
```

Where:
- (x_COM, y_COM, z_COM) is the center of mass position
- (ẍ_COM, ÿ_COM) is the center of mass acceleration
- g is gravitational acceleration
- z_support is the height of the support surface

### ZMP-Based Balance Control

```python
class ZMPBalanceController:
    def __init__(self):
        # Robot parameters
        self.mass = 60.0  # kg
        self.gravity = 9.81  # m/s^2
        self.com_height = 0.8  # m (typical for adult-sized humanoid)
        
        # Control parameters
        self.kp_zmp = 50.0  # Proportional gain for ZMP control
        self.kd_zmp = 10.0  # Derivative gain for ZMP control
        self.kp_com = 10.0  # Proportional gain for CoM control
        self.kd_com = 5.0   # Derivative gain for CoM control
        
        # Support polygon (approximate rectangle for single support)
        self.support_polygon = np.array([
            [-0.1, -0.05],  # front left
            [0.1, -0.05],   # front right
            [0.1, 0.05],    # back right
            [-0.1, 0.05]    # back left
        ])
        
        # Previous values for derivative calculation
        self.prev_zmp_error = np.zeros(2)
        self.prev_com_error = np.zeros(3)
        self.prev_time = None
        
        # Trajectory planning
        self.desired_zmp_trajectory = []
        self.desired_com_trajectory = []
    
    def calculate_zmp(self, com_pos, com_acc):
        """
        Calculate ZMP from CoM position and acceleration
        """
        x_com, y_com, z_com = com_pos
        x_acc, y_acc, z_acc = com_acc
        
        # ZMP calculation (simplified for level ground)
        zmp_x = x_com - (z_com / self.gravity) * x_acc
        zmp_y = y_com - (z_com / self.gravity) * y_acc
        
        return np.array([zmp_x, zmp_y])
    
    def is_zmp_stable(self, zmp, margin=0.02):
        """
        Check if ZMP is within support polygon with a safety margin
        """
        # Expand support polygon inward by margin
        adjusted_polygon = self.expand_polygon(self.support_polygon, -margin)
        
        return self.point_in_polygon(zmp, adjusted_polygon)
    
    def expand_polygon(self, polygon, offset):
        """
        Expand or contract a polygon by a given offset
        """
        # This is a simplified implementation
        # In practice, you'd use proper polygon offsetting algorithms
        centroid = np.mean(polygon, axis=0)
        expanded = []
        
        for point in polygon:
            direction = point - centroid
            length = np.linalg.norm(direction)
            if length > 0:
                normalized = direction / length
                expanded_point = point + normalized * offset
            else:
                expanded_point = point
            expanded.append(expanded_point)
        
        return np.array(expanded)
    
    def point_in_polygon(self, point, polygon):
        """
        Check if a 2D point is inside a polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def calculate_balance_correction(self, current_zmp, desired_zmp, current_com, current_com_vel):
        """
        Calculate balance correction based on ZMP error
        """
        # Calculate ZMP error
        zmp_error = desired_zmp - current_zmp
        
        # Calculate time step
        current_time = time.time()
        if self.prev_time is not None:
            dt = current_time - self.prev_time
        else:
            dt = 0.01  # Default time step
        
        self.prev_time = current_time
        
        # Calculate derivative of ZMP error
        if dt > 0:
            zmp_error_derivative = (zmp_error - self.prev_zmp_error) / dt
        else:
            zmp_error_derivative = np.zeros_like(zmp_error)
        
        # PID control for ZMP
        zmp_correction = (self.kp_zmp * zmp_error + 
                         self.kd_zmp * zmp_error_derivative)
        
        # Apply limits to correction
        max_correction = 0.1  # meters
        zmp_correction = np.clip(zmp_correction, -max_correction, max_correction)
        
        # Update previous error
        self.prev_zmp_error = zmp_error
        
        # Calculate CoM correction to achieve desired ZMP
        com_correction = self.calculate_com_correction_for_zmp(zmp_correction, current_com)
        
        return zmp_correction, com_correction
    
    def calculate_com_correction_for_zmp(self, zmp_correction, current_com):
        """
        Calculate CoM adjustment needed to correct ZMP
        """
        # Simplified relationship: adjust CoM proportionally to ZMP error
        # In reality, this would involve more complex dynamics
        com_correction = np.zeros(3)
        com_correction[0] = zmp_correction[0] * 0.8  # Map x ZMP error to x CoM adjustment
        com_correction[1] = zmp_correction[1] * 0.8  # Map y ZMP error to y CoM adjustment
        # z component typically kept constant for balance
        
        return com_correction
    
    def generate_zmp_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """
        Generate a ZMP trajectory from start to end position
        """
        num_steps = int(duration / dt)
        trajectory = []
        
        for i in range(num_steps):
            t = i / (num_steps - 1) if num_steps > 1 else 0
            pos = start_pos + t * (end_pos - start_pos)
            trajectory.append(pos)
        
        return trajectory
    
    def update_support_polygon(self, left_foot_pos, right_foot_pos, foot_size=0.15):
        """
        Update support polygon based on foot positions
        """
        if left_foot_pos is not None and right_foot_pos is not None:
            # Double support - create polygon encompassing both feet
            self.support_polygon = self.calculate_support_polygon(left_foot_pos, right_foot_pos, foot_size)
        elif left_foot_pos is not None:
            # Left foot support
            self.support_polygon = self.calculate_single_foot_polygon(left_foot_pos, foot_size)
        elif right_foot_pos is not None:
            # Right foot support
            self.support_polygon = self.calculate_single_foot_polygon(right_foot_pos, foot_size)
        else:
            # No support - set to empty polygon
            self.support_polygon = np.array([])
    
    def calculate_support_polygon(self, left_foot_pos, right_foot_pos, foot_size):
        """
        Calculate support polygon for double support
        """
        # Create convex hull of both feet
        left_points = self.calculate_single_foot_polygon(left_foot_pos, foot_size)
        right_points = self.calculate_single_foot_polygon(right_foot_pos, foot_size)
        
        all_points = np.vstack([left_points, right_points])
        
        # Compute convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(all_points)
        return all_points[hull.vertices]
    
    def calculate_single_foot_polygon(self, foot_pos, foot_size):
        """
        Calculate rectangular polygon for a single foot
        """
        x, y, _ = foot_pos
        half_size = foot_size / 2.0
        
        return np.array([
            [x - half_size, y - half_size],
            [x + half_size, y - half_size],
            [x + half_size, y + half_size],
            [x - half_size, y + half_size]
        ])
    
    def stabilize_balance(self, current_com_pos, current_com_vel, current_com_acc,
                         left_foot_pos=None, right_foot_pos=None):
        """
        Main balance stabilization function
        """
        # Update support polygon based on foot positions
        self.update_support_polygon(left_foot_pos, right_foot_pos)
        
        # Calculate current ZMP
        current_zmp = self.calculate_zmp(current_com_pos, current_com_acc)
        
        # Determine desired ZMP (typically center of support polygon when stable)
        if len(self.support_polygon) > 0:
            desired_zmp = np.mean(self.support_polygon, axis=0)
        else:
            desired_zmp = current_zmp  # Stay at current position if no support
        
        # Calculate balance corrections
        zmp_corr, com_corr = self.calculate_balance_correction(
            current_zmp, desired_zmp, current_com_pos, current_com_vel
        )
        
        # Check stability
        is_stable = self.is_zmp_stable(current_zmp)
        
        # Return correction values and stability status
        return {
            'zmp_correction': zmp_corr,
            'com_correction': com_corr,
            'current_zmp': current_zmp,
            'desired_zmp': desired_zmp,
            'is_stable': is_stable,
            'support_polygon': self.support_polygon
        }

# Example usage
import time

zmp_controller = ZMPBalanceController()

# Example CoM state
current_com_pos = np.array([0.0, 0.0, 0.8])
current_com_vel = np.array([0.01, 0.0, 0.0])
current_com_acc = np.array([0.001, 0.0, 0.0])

# Example foot positions
left_foot_pos = np.array([0.1, 0.15, 0.0])
right_foot_pos = np.array([0.1, -0.15, 0.0])

# Perform balance stabilization
balance_result = zmp_controller.stabilize_balance(
    current_com_pos, current_com_vel, current_com_acc,
    left_foot_pos, right_foot_pos
)

print("Balance Stabilization Result:")
print(f"ZMP Correction: {balance_result['zmp_correction']}")
print(f"CoM Correction: {balance_result['com_correction']}")
print(f"Current ZMP: {balance_result['current_zmp']}")
print(f"Desired ZMP: {balance_result['desired_zmp']}")
print(f"Is Stable: {balance_result['is_stable']}")
```

## Torque and joint control

Controlling the torques applied to each joint is critical for achieving stable and efficient movement in humanoid robots. This involves understanding the dynamics of the robot and applying appropriate control strategies.

### Robot Dynamics

The dynamics of a robot are described by the equation:

```
M(q)q̈ + C(q, q̇)q̇ + g(q) = τ
```

Where:
- M(q) is the mass/inertia matrix
- C(q, q̇)q̇ represents Coriolis and centrifugal forces
- g(q) represents gravitational forces
- τ represents applied joint torques
- q, q̇, q̈ are joint positions, velocities, and accelerations

### Joint Control Implementation

```python
class JointController:
    def __init__(self, num_joints):
        self.num_joints = num_joints
        
        # PID control gains for each joint
        self.kp = np.ones(num_joints) * 100.0  # Proportional gain
        self.ki = np.ones(num_joints) * 0.1    # Integral gain  
        self.kd = np.ones(num_joints) * 10.0   # Derivative gain
        
        # Joint limits
        self.joint_limits = {
            'min': np.ones(num_joints) * -np.pi,
            'max': np.ones(num_joints) * np.pi
        }
        
        # Torque limits
        self.torque_limits = np.ones(num_joints) * 100.0  # Nm
        
        # Control history for integral term
        self.error_history = np.zeros(num_joints)
        self.prev_error = np.zeros(num_joints)
        
        # Gravity compensation parameters
        self.gravity = 9.81
        self.link_masses = np.ones(num_joints) * 1.0  # Simplified masses
        self.link_lengths = np.ones(num_joints) * 0.1  # Simplified lengths
        
        # Friction compensation
        self.friction_static = np.ones(num_joints) * 2.0   # Static friction (Nm)
        self.friction_viscous = np.ones(num_joints) * 0.5  # Viscous friction (Nm/(rad/s))
    
    def compute_control_torques(self, desired_pos, desired_vel, desired_acc,
                               current_pos, current_vel, dt=0.001):
        """
        Compute control torques using PID control with feedforward terms
        """
        # Calculate position and velocity errors
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel
        
        # PID control with feedforward acceleration term
        proportional_term = self.kp * pos_error
        integral_term = self.ki * (self.error_history + pos_error * dt)
        derivative_term = self.kd * ((pos_error - self.prev_error) / dt if dt > 0 else 0)
        
        # PID contribution
        pid_torques = proportional_term + integral_term + derivative_term
        
        # Feedforward term for desired acceleration
        feedforward_torques = self.estimate_feedforward_torques(desired_acc, desired_vel)
        
        # Gravity compensation
        gravity_compensation = self.compensate_gravity(current_pos)
        
        # Friction compensation
        friction_compensation = self.compensate_friction(current_vel)
        
        # Total control torques
        total_torques = pid_torques + feedforward_torques + gravity_compensation + friction_compensation
        
        # Apply torque limits
        total_torques = np.clip(total_torques, -self.torque_limits, self.torque_limits)
        
        # Update history
        self.error_history = self.error_history + pos_error * dt
        self.prev_error = pos_error
        
        return total_torques
    
    def estimate_feedforward_torques(self, desired_acc, desired_vel):
        """
        Estimate feedforward torques based on desired acceleration and velocity
        """
        # This is a simplified model - in practice, you'd use full dynamics
        # M(q)q̈ + C(q, q̇)q̇ + g(q) = τ
        # We approximate: τ_ff ≈ M*q̈_d + C*q̇_d
        # where M is diagonal matrix of apparent inertia
        # and C includes Coriolis and viscous damping
        
        # Simplified feedforward: mainly acceleration term
        inertia_term = 0.5 * desired_acc  # Simplified inertia matrix
        coriolis_term = 0.1 * desired_vel * np.abs(desired_vel)  # Simplified Coriolis
        
        return inertia_term + coriolis_term
    
    def compensate_gravity(self, joint_pos):
        """
        Compensate for gravitational forces
        """
        # Calculate gravity-compensating torques
        # This is a simplified model - assumes simple pendulum approximation
        gravity_torques = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            # Simple gravity compensation: tau = m*g*l*cos(theta)
            # where m is mass, l is link length, theta is joint angle
            gravity_torques[i] = (self.link_masses[i] * self.gravity * 
                                self.link_lengths[i] * np.cos(joint_pos[i]))
        
        return gravity_torques
    
    def compensate_friction(self, joint_vel):
        """
        Compensate for friction forces
        """
        friction_torques = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            # Static friction: sign(v) * F_static if |v| is small, else 0
            if abs(joint_vel[i]) < 0.01:  # Near zero velocity
                friction_torques[i] = np.sign(joint_vel[i]) * self.friction_static[i]
            else:
                # Viscous friction: -F_viscous * v
                friction_torques[i] = -self.friction_viscous[i] * joint_vel[i]
        
        return friction_torques
    
    def compute_inverse_dynamics(self, joint_pos, joint_vel, joint_acc):
        """
        Compute required torques using inverse dynamics
        This is a simplified implementation
        """
        # Full inverse dynamics: τ = M(q)q̈ + C(q, q̇)q̇ + g(q)
        # where M is mass matrix, C contains Coriolis terms, g is gravity
        
        # Calculate mass matrix (simplified diagonal)
        M = self.compute_mass_matrix(joint_pos)
        
        # Calculate Coriolis and centrifugal terms (simplified)
        C = self.compute_coriolis_matrix(joint_pos, joint_vel)
        
        # Calculate gravity terms
        g = self.compute_gravity_vector(joint_pos)
        
        # Full inverse dynamics calculation
        tau = (M @ joint_acc) + (C @ joint_vel) + g
        
        return tau
    
    def compute_mass_matrix(self, joint_pos):
        """
        Compute the mass/inertia matrix M(q)
        This is a simplified diagonal approximation
        """
        # In a real implementation, this would involve complex calculations
        # based on the robot's kinematic structure
        M = np.zeros((self.num_joints, self.num_joints))
        
        # Simplified: diagonal matrix with position-dependent terms
        for i in range(self.num_joints):
            # Approximate moment of inertia for each link
            M[i, i] = (self.link_masses[i] * self.link_lengths[i]**2 * 
                      (1 + 0.1 * np.sin(joint_pos[i])**2))  # Position-dependent factor
        
        return M
    
    def compute_coriolis_matrix(self, joint_pos, joint_vel):
        """
        Compute Coriolis and centrifugal matrix C(q, q̇)
        """
        # Simplified Coriolis matrix calculation
        C = np.zeros((self.num_joints, self.num_joints))
        
        # Off-diagonal terms that couple joint motions
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    # Simplified coupling term
                    C[i, j] = (0.1 * self.link_masses[i] * self.link_lengths[i] * 
                              self.link_lengths[j] * joint_vel[j] * np.cos(joint_pos[i] - joint_pos[j]))
        
        # Diagonal terms
        for i in range(self.num_joints):
            C[i, i] = 0.05 * self.link_masses[i] * self.link_lengths[i]**2 * joint_vel[i]
        
        return C
    
    def compute_gravity_vector(self, joint_pos):
        """
        Compute gravity vector g(q)
        """
        g_vec = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            # Gravity effect varies with joint angle
            g_vec[i] = (self.link_masses[i] * self.gravity * self.link_lengths[i] * 
                       np.cos(joint_pos[i]))
        
        return g_vec
    
    def safety_check(self, torques, joint_pos, joint_vel):
        """
        Perform safety checks on computed torques
        """
        # Check for excessive torques
        if np.any(np.abs(torques) > self.torque_limits * 0.9):
            print("WARNING: Torque limits approaching!")
        
        # Check for joint limits
        if np.any(joint_pos < self.joint_limits['min']) or np.any(joint_pos > self.joint_limits['max']):
            print("WARNING: Joint limits exceeded!")
        
        # Check for excessive velocities
        max_vel = 5.0  # rad/s
        if np.any(np.abs(joint_vel) > max_vel):
            print("WARNING: Joint velocities excessive!")
        
        return True  # All safety checks passed

# Example usage of joint controller
joint_ctrl = JointController(6)  # 6-DOF arm

# Example trajectory
desired_pos = np.array([0.5, 0.2, 0.3, -1.0, 0.1, 0.05])
desired_vel = np.array([0.1, 0.05, 0.05, -0.2, 0.02, 0.01])
desired_acc = np.array([0.01, 0.005, 0.005, -0.02, 0.002, 0.001])

current_pos = np.array([0.4, 0.15, 0.25, -0.9, 0.08, 0.03])
current_vel = np.array([0.08, 0.04, 0.04, -0.15, 0.01, 0.005])

dt = 0.001
torques = joint_ctrl.compute_control_torques(
    desired_pos, desired_vel, desired_acc,
    current_pos, current_vel, dt
)

print(f"Computed joint torques: {torques}")

# Check safety
joint_ctrl.safety_check(torques, current_pos, current_vel)
```

## Conclusion

Kinematics and dynamics form the foundation of humanoid robot control. Forward kinematics allows us to determine end-effector positions from joint angles, while inverse kinematics solves the reverse problem. Dynamic balance, particularly through ZMP theory, is essential for stable locomotion. Proper torque control ensures smooth and safe movement while respecting physical constraints.

The integration of these concepts enables humanoid robots to perform complex movements while maintaining stability. As humanoid robotics continues to advance, these fundamental principles remain crucial for developing more capable and human-like robots.