---
title: Bipedal Locomotion
sidebar_position: 2
description: Walking gaits, balance restoration, leg trajectory planning, and fall prevention for humanoid robots
---

# Bipedal Locomotion

## Walking gaits

Bipedal locomotion is one of the most challenging aspects of humanoid robotics, requiring precise control of multiple degrees of freedom to achieve stable, efficient, and human-like walking. The study of walking gaits involves understanding the biomechanics of human walking and translating these principles into robotic control systems.

### Basic Gait Terminology

Understanding the terminology of human gait is essential for developing robotic walking controllers:

- **Step**: The distance between two consecutive contacts of the same foot
- **Stride**: The distance between two consecutive contacts of the same foot (two steps)
- **Stance Phase**: Period when the foot is in contact with the ground
- **Swing Phase**: Period when the foot is off the ground and swinging forward
- **Double Support Phase**: Period when both feet are in contact with the ground
- **Single Support Phase**: Period when only one foot is in contact with the ground

### Common Walking Gaits

#### Static Walking
In static walking, the robot maintains static stability throughout the walking cycle. This means the center of mass (CoM) projection always remains within the support polygon formed by the feet.

```python
class StaticWalkGait:
    def __init__(self):
        self.step_length = 0.2  # meters
        self.step_width = 0.3  # meters (distance between feet)
        self.step_height = 0.05  # meters (foot clearance)
        self.nominal_com_height = 0.8  # meters
        
        # Timing parameters
        self.step_duration = 2.0  # seconds per step
        self.double_support_ratio = 0.2  # 20% of step time in double support
        
    def generate_trajectory(self, num_steps):
        """
        Generate static walking trajectory
        """
        trajectories = {
            'left_foot': [],
            'right_foot': [],
            'com': [],
            'zmp': []
        }
        
        # Starting positions
        left_foot_pos = [0, self.step_width/2, 0]
        right_foot_pos = [0, -self.step_width/2, 0]
        com_pos = [0, 0, self.nominal_com_height]
        
        for step in range(num_steps):
            # Determine which foot is swing foot (odd steps: right foot swings, even: left foot swings)
            if step % 2 == 0:  # Even steps: left foot is stance, right foot swings
                stance_foot = 'left'
                swing_foot = 'right'
                next_swing_pos = [left_foot_pos[0] + self.step_length, -self.step_width/2, 0]
            else:  # Odd steps: right foot is stance, left foot swings
                stance_foot = 'right'
                swing_foot = 'left'
                next_swing_pos = [right_foot_pos[0] + self.step_length, self.step_width/2, 0]
            
            # Generate trajectory for this step
            step_trajectory = self.generate_single_step(
                stance_foot, swing_foot, 
                left_foot_pos, right_foot_pos, 
                next_swing_pos, com_pos
            )
            
            # Append to overall trajectories
            for key in trajectories:
                trajectories[key].extend(step_trajectory[key])
        
        return trajectories
    
    def generate_single_step(self, stance_foot, swing_foot, 
                           left_pos, right_pos, next_swing_pos, com_pos):
        """
        Generate trajectory for a single step
        """
        step_trajectory = {
            'left_foot': [],
            'right_foot': [],
            'com': [],
            'zmp': []
        }
        
        # Calculate step timing
        double_support_time = self.step_duration * self.double_support_ratio
        single_support_time = self.step_duration - double_support_time
        dt = 0.01  # 10ms control cycle
        
        # Phase 1: Double support (beginning of step)
        for t in np.arange(0, double_support_time, dt):
            # CoM moves toward new support foot
            progress = t / double_support_time
            new_com_x = com_pos[0] + (next_swing_pos[0] - com_pos[0]) * progress * 0.3  # Move 30% of the way
            
            # Both feet stay in place
            step_trajectory['left_foot'].append(left_pos.copy())
            step_trajectory['right_foot'].append(right_pos.copy())
            step_trajectory['com'].append([new_com_x, com_pos[1], com_pos[2]])
            
            # Calculate ZMP (simplified)
            zmp_x = new_com_x  # In static walking, ZMP approximately equals CoM
            zmp_y = com_pos[1]
            step_trajectory['zmp'].append([zmp_x, zmp_y])
        
        # Phase 2: Single support (swing phase)
        for t in np.arange(0, single_support_time, dt):
            progress = t / single_support_time
            swing_phase = self.calculate_swing_trajectory(
                left_pos if stance_foot == 'left' else right_pos,
                next_swing_pos, progress
            )
            
            # Update positions based on which foot is swing foot
            if swing_foot == 'left':
                step_trajectory['left_foot'].append(swing_phase)
                step_trajectory['right_foot'].append(right_pos.copy())
            else:
                step_trajectory['left_foot'].append(left_pos.copy())
                step_trajectory['right_foot'].append(swing_phase)
            
            # CoM continues to move toward new support position
            com_progress = double_support_ratio + progress * (1 - double_support_ratio)
            new_com_x = com_pos[0] + (next_swing_pos[0] - com_pos[0]) * com_progress
            new_com_y = com_pos[1] + (next_swing_pos[1] - com_pos[1]) * com_progress * 0.1  # Small lateral movement
            
            step_trajectory['com'].append([new_com_x, new_com_y, com_pos[2]])
            
            # Calculate ZMP
            zmp_x = new_com_x
            zmp_y = new_com_y
            step_trajectory['zmp'].append([zmp_x, zmp_y])
        
        # Phase 3: Double support (end of step)
        for t in np.arange(0, double_support_time, dt):
            progress = t / double_support_time
            # CoM continues to move toward new support foot
            final_com_x = next_swing_pos[0]
            final_com_y = next_swing_pos[1]
            
            new_com_x = step_trajectory['com'][-1][0] + (final_com_x - step_trajectory['com'][-1][0]) * progress
            new_com_y = step_trajectory['com'][-1][1] + (final_com_y - step_trajectory['com'][-1][1]) * progress * 0.1
            
            # Both feet stay in place (new positions)
            step_trajectory['left_foot'].append(left_pos.copy() if stance_foot == 'left' else next_swing_pos.copy())
            step_trajectory['right_foot'].append(right_pos.copy() if stance_foot == 'right' else next_swing_pos.copy())
            step_trajectory['com'].append([new_com_x, new_com_y, com_pos[2]])
            
            # Calculate ZMP
            zmp_x = new_com_x
            zmp_y = new_com_y
            step_trajectory['zmp'].append([zmp_x, zmp_y])
        
        return step_trajectory
    
    def calculate_swing_trajectory(self, start_pos, end_pos, progress):
        """
        Calculate smooth swing trajectory with foot clearance
        """
        # Linear interpolation with parabolic vertical component for foot clearance
        x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
        y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
        z = start_pos[2]  # Initially on ground
        
        # Add parabolic trajectory for foot clearance
        # Use 5th order polynomial for smooth lift and landing
        if progress < 0.5:
            # Lifting phase
            lift_progress = progress * 2  # Scale to 0-1
            z_lift = self.step_height * (10*lift_progress**3 - 15*lift_progress**4 + 6*lift_progress**5)
            z = start_pos[2] + z_lift
        else:
            # Landing phase
            land_progress = (progress - 0.5) * 2  # Scale to 0-1
            z_lift = self.step_height * (10*land_progress**3 - 15*land_progress**4 + 6*land_progress**5)
            z = end_pos[2] + self.step_height - z_lift
        
        return [x, y, max(z, end_pos[2])]  # Don't go below ground level

# Example usage
static_gait = StaticWalkGait()
trajectories = static_gait.generate_trajectory(4)  # Generate 4 steps
print(f"Generated {len(trajectories['com'])} trajectory points for 4 steps")
```

#### Dynamic Walking

Dynamic walking allows the robot to have periods where the center of mass projection moves outside the support polygon, similar to human walking. This results in more natural and efficient locomotion.

```python
class DynamicWalkGait:
    def __init__(self):
        # Walking parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.25  # meters
        self.nominal_com_height = 0.75  # meters
        self.step_duration = 0.8  # seconds per step (faster than static walking)
        
        # Inverted pendulum parameters
        self.pendulum_length = self.nominal_com_height  # Length of inverted pendulum
        self.gravity = 9.81
        
        # Timing ratios
        self.double_support_ratio = 0.1  # Shorter double support for dynamic walking
        
        # Gait pattern parameters
        self.com_oscillation_amplitude = 0.02  # Lateral CoM oscillation
        self.com_forward_speed = self.step_length / self.step_duration  # Nominal forward speed
    
    def generate_dynamic_trajectory(self, num_steps):
        """
        Generate dynamic walking trajectory using inverted pendulum model
        """
        trajectories = {
            'left_foot': [],
            'right_foot': [],
            'com': [],
            'zmp': [],
            'timestamps': []
        }
        
        # Starting positions
        left_foot_pos = [0, self.step_width/2, 0]
        right_foot_pos = [0, -self.step_width/2, 0]
        com_pos = [0, 0, self.nominal_com_height]
        
        current_time = 0.0
        
        for step in range(num_steps):
            # Determine support foot and swing foot
            if step % 2 == 0:  # Even steps: right foot swings forward
                support_foot = 'left'
                swing_foot = 'right'
                next_swing_pos = [left_foot_pos[0] + self.step_length, -self.step_width/2, 0]
            else:  # Odd steps: left foot swings forward
                support_foot = 'right'
                swing_foot = 'left'
                next_swing_pos = [right_foot_pos[0] + self.step_length, self.step_width/2, 0]
            
            # Generate step trajectory
            step_data = self.generate_dynamic_step(
                support_foot, swing_foot,
                left_foot_pos, right_foot_pos,
                next_swing_pos, com_pos,
                current_time
            )
            
            # Append to trajectories
            for key in trajectories:
                if key != 'timestamps':
                    trajectories[key].extend(step_data[key])
                else:
                    trajectories[key].extend(step_data['time'])
            
            # Update positions for next step
            if step % 2 == 0:
                right_foot_pos = next_swing_pos
            else:
                left_foot_pos = next_swing_pos
            
            # Update CoM position
            com_pos[0] = step_data['com'][-1][0]
            com_pos[1] = step_data['com'][-1][1]
            com_pos[2] = step_data['com'][-1][2]
            
            current_time = step_data['time'][-1]
        
        return trajectories
    
    def generate_dynamic_step(self, support_foot, swing_foot,
                            left_pos, right_pos, next_swing_pos, com_pos, start_time):
        """
        Generate trajectory for a single dynamic walking step
        """
        step_data = {
            'left_foot': [],
            'right_foot': [],
            'com': [],
            'zmp': [],
            'time': []
        }
        
        dt = 0.01  # 10ms control cycle
        step_time = self.step_duration
        
        # Calculate key times
        double_support_time = step_time * self.double_support_ratio
        single_support_time = step_time - 2 * double_support_time  # Two double support phases
        
        current_time = start_time
        
        # Phase 1: Beginning double support
        for t in np.arange(0, double_support_time, dt):
            # CoM begins to shift toward new support foot
            progress = t / double_support_time
            shift_amount = 0.3  # Only shift 30% initially
            
            new_com_x = com_pos[0] + (next_swing_pos[0] - com_pos[0]) * progress * shift_amount
            new_com_y = com_pos[1] + (next_swing_pos[1] - com_pos[1]) * progress * shift_amount * 0.5  # Reduced lateral shift
            
            # Both feet stay in place
            step_data['left_foot'].append(left_pos.copy())
            step_data['right_foot'].append(right_pos.copy())
            step_data['com'].append([new_com_x, new_com_y, com_pos[2]])
            
            # Calculate ZMP based on inverted pendulum model
            zmp = self.calculate_zmp_inverted_pendulum([new_com_x, new_com_y, com_pos[2]], [0, 0, 0], [0, 0, 0])
            step_data['zmp'].append(zmp)
            step_data['time'].append(current_time)
            
            current_time += dt
        
        # Phase 2: Single support (swing phase)
        for t in np.arange(0, single_support_time, dt):
            progress = t / single_support_time
            
            # Swing foot trajectory
            swing_pos = self.calculate_swing_trajectory(
                left_pos if swing_foot == 'left' else right_pos,
                next_swing_pos, progress
            )
            
            # Update feet positions
            if swing_foot == 'left':
                step_data['left_foot'].append(swing_pos)
                step_data['right_foot'].append(right_pos.copy())
            else:
                step_data['left_foot'].append(left_pos.copy())
                step_data['right_foot'].append(swing_pos)
            
            # CoM follows inverted pendulum dynamics
            # Calculate CoM position based on desired ZMP
            zmp_x = self.calculate_desired_zmp_x(com_pos[0], next_swing_pos[0], progress)
            zmp_y = self.calculate_desired_zmp_y(com_pos[1], next_swing_pos[1], progress)
            
            # Calculate CoM position from ZMP using inverted pendulum model
            com_x, com_y = self.calculate_com_from_zmp(zmp_x, zmp_y, com_pos[2])
            
            step_data['com'].append([com_x, com_y, com_pos[2]])
            
            # Calculate actual ZMP
            zmp = self.calculate_zmp_inverted_pendulum([com_x, com_y, com_pos[2]], [0, 0, 0], [0, 0, 0])
            step_data['zmp'].append(zmp)
            step_data['time'].append(current_time)
            
            current_time += dt
        
        # Phase 3: Ending double support
        for t in np.arange(0, double_support_time, dt):
            progress = t / double_support_time
            
            # Complete CoM shift
            final_com_x = next_swing_pos[0]
            final_com_y = next_swing_pos[1]
            
            new_com_x = step_data['com'][-1][0] + (final_com_x - step_data['com'][-1][0]) * progress
            new_com_y = step_data['com'][-1][1] + (final_com_y - step_data['com'][-1][1]) * progress * 0.3  # Reduced lateral shift
            
            # Both feet at final positions
            step_data['left_foot'].append(left_pos.copy() if swing_foot == 'right' else next_swing_pos.copy())
            step_data['right_foot'].append(right_pos.copy() if swing_foot == 'left' else next_swing_pos.copy())
            step_data['com'].append([new_com_x, new_com_y, com_pos[2]])
            
            # Calculate ZMP
            zmp = self.calculate_zmp_inverted_pendulum([new_com_x, new_com_y, com_pos[2]], [0, 0, 0], [0, 0, 0])
            step_data['zmp'].append(zmp)
            step_data['time'].append(current_time)
            
            current_time += dt
        
        return step_data
    
    def calculate_desired_zmp_x(self, start_x, end_x, progress):
        """
        Calculate desired ZMP x position based on walking pattern
        """
        # In dynamic walking, ZMP leads the CoM to maintain balance
        # Use a pattern that moves ZMP to the new support foot
        return start_x + (end_x - start_x) * progress
    
    def calculate_desired_zmp_y(self, start_y, end_y, progress):
        """
        Calculate desired ZMP y position based on walking pattern
        """
        # For lateral movement, ZMP moves gradually
        return start_y + (end_y - start_y) * progress * 0.5  # Slower lateral shift
    
    def calculate_com_from_zmp(self, zmp_x, zmp_y, com_height):
        """
        Calculate CoM position from ZMP using inverted pendulum model
        """
        # Simplified relationship: CoM slightly leads ZMP for dynamic balance
        com_x = zmp_x + 0.02  # CoM slightly ahead of ZMP
        com_y = zmp_y + 0.01  # Small lateral offset
        
        return com_x, com_y
    
    def calculate_zmp_inverted_pendulum(self, com_pos, com_vel, com_acc):
        """
        Calculate ZMP based on inverted pendulum model
        """
        x_com, y_com, z_com = com_pos
        x_vel, y_vel, z_vel = com_vel
        x_acc, y_acc, z_acc = com_acc
        
        # ZMP calculation for inverted pendulum
        zmp_x = x_com - (z_com / self.gravity) * x_acc
        zmp_y = y_com - (z_com / self.gravity) * y_acc
        
        return [zmp_x, zmp_y]
```

### Walk Pattern Generators

Walk pattern generators create the rhythmic patterns needed for stable locomotion.

```python
class WalkPatternGenerator:
    def __init__(self):
        # Gait parameters
        self.stride_length = 0.3  # meters
        self.stride_width = 0.25  # meters
        self.step_height = 0.08  # meters
        self.com_height = 0.8  # meters
        self.step_period = 0.8  # seconds
        
        # Oscillator parameters for rhythmic control
        self.oscillator_freq = 1.0 / self.step_period  # Frequency in Hz
        self.phase_diff_stance_swing = np.pi  # 180° phase difference
        
        # Trajectory generation parameters
        self.traj_dt = 0.01  # 10ms discretization
        
        # Interpolation parameters
        self.swing_lift_ratio = 0.3  # When in swing phase to lift foot
        self.swing_land_ratio = 0.7  # When in swing phase to start lowering foot
    
    def generate_walk_pattern(self, duration, walking_speed=0.3):
        """
        Generate complete walking pattern for a given duration
        """
        total_steps = int(duration / self.step_period)
        
        # Calculate adjusted parameters based on speed
        adjusted_stride = self.stride_length * (walking_speed / 0.3)  # Normalize to 0.3 m/s
        
        # Generate trajectories for each step
        pattern = {
            'time': [],
            'left_foot': {'x': [], 'y': [], 'z': []},
            'right_foot': {'x': [], 'y': [], 'z': []},
            'com': {'x': [], 'y': [], 'z': []},
            'zmp': {'x': [], 'y': []},
            'phase': []  # 'double_support', 'left_stance', 'right_stance'
        }
        
        # Starting positions
        left_foot_x, left_foot_y = 0, self.stride_width / 2
        right_foot_x, right_foot_y = 0, -self.stride_width / 2
        com_x, com_y = 0, 0
        
        current_time = 0.0
        
        for step in range(total_steps):
            # Determine step type (left or right swing)
            step_type = 'left_swing' if step % 2 == 0 else 'right_swing'
            
            # Generate single step pattern
            step_pattern = self.generate_single_step_pattern(
                step_type, current_time, adjusted_stride,
                left_foot_x, left_foot_y, right_foot_x, right_foot_y,
                com_x, com_y
            )
            
            # Append to main pattern
            for key in pattern:
                if key == 'time':
                    pattern['time'].extend(step_pattern['time'])
                elif key == 'phase':
                    pattern['phase'].extend(step_pattern['phase'])
                elif key in ['left_foot', 'right_foot', 'com', 'zmp']:
                    for coord in ['x', 'y', 'z'] if key != 'zmp' else ['x', 'y']:
                        if coord in step_pattern[key]:
                            pattern[key][coord].extend(step_pattern[key][coord])
            
            # Update starting positions for next step
            if step_type == 'left_swing':
                left_foot_x = right_foot_x + adjusted_stride
                com_x = right_foot_x + adjusted_stride / 2
            else:
                right_foot_x = left_foot_x + adjusted_stride
                com_x = left_foot_x + adjusted_stride / 2
            
            current_time = step_pattern['time'][-1]
        
        return pattern
    
    def generate_single_step_pattern(self, step_type, start_time, stride_length,
                                   left_x, left_y, right_x, right_y, com_x, com_y):
        """
        Generate pattern for a single step
        """
        step_pattern = {
            'time': [],
            'left_foot': {'x': [], 'y': [], 'z': []},
            'right_foot': {'x': [], 'y': [], 'z': []},
            'com': {'x': [], 'y': [], 'z': []},
            'zmp': {'x': [], 'y': []},
            'phase': []
        }
        
        # Calculate phase durations
        double_support_dur = self.step_period * 0.1  # 10% double support
        single_support_dur = self.step_period - 2 * double_support_dur  # Remaining for single support
        
        current_time = start_time
        
        # Phase 1: Initial double support
        for t in np.arange(0, double_support_dur, self.traj_dt):
            time_point = start_time + t
            
            # Both feet stay in place initially
            if step_type == 'left_swing':
                # Right foot is stance, left foot will swing
                step_pattern['left_foot']['x'].append(left_x)
                step_pattern['left_foot']['y'].append(left_y)
                step_pattern['left_foot']['z'].append(0.0)
                
                step_pattern['right_foot']['x'].append(right_x)
                step_pattern['right_foot']['y'].append(right_y)
                step_pattern['right_foot']['z'].append(0.0)
            else:
                # Left foot is stance, right foot will swing
                step_pattern['left_foot']['x'].append(left_x)
                step_pattern['left_foot']['y'].append(left_y)
                step_pattern['left_foot']['z'].append(0.0)
                
                step_pattern['right_foot']['x'].append(right_x)
                step_pattern['right_foot']['y'].append(right_y)
                step_pattern['right_foot']['z'].append(0.0)
            
            # CoM begins shifting
            shift_progress = t / double_support_dur
            target_com_x = (left_x + right_x) / 2 + stride_length / 2  # Move toward new support
            new_com_x = com_x + (target_com_x - com_x) * shift_progress
            
            step_pattern['com']['x'].append(new_com_x)
            step_pattern['com']['y'].append(0.0)  # Average of foot positions
            step_pattern['com']['z'].append(self.com_height)
            
            # Calculate ZMP (simplified)
            step_pattern['zmp']['x'].append(new_com_x)
            step_pattern['zmp']['y'].append(0.0)
            
            step_pattern['phase'].append('double_support')
            step_pattern['time'].append(time_point)
        
        # Phase 2: Single support (swing phase)
        for t in np.arange(0, single_support_dur, self.traj_dt):
            time_point = start_time + double_support_dur + t
            swing_progress = t / single_support_dur
            
            # Calculate swing foot trajectory
            if step_type == 'left_swing':
                # Left foot swings forward
                new_left_x = self.interpolate_swing_x(left_x, left_x + stride_length, swing_progress)
                new_left_y = self.interpolate_swing_y(left_y, -self.stride_width/2, swing_progress)
                new_left_z = self.interpolate_swing_z(swing_progress)
                
                # Update trajectories
                step_pattern['left_foot']['x'].append(new_left_x)
                step_pattern['left_foot']['y'].append(new_left_y)
                step_pattern['left_foot']['z'].append(new_left_z)
                
                # Right foot stays in place
                step_pattern['right_foot']['x'].append(right_x)
                step_pattern['right_foot']['y'].append(right_y)
                step_pattern['right_foot']['z'].append(0.0)
                
                # Phase label
                step_pattern['phase'].append('right_stance')
            else:
                # Right foot swings forward
                new_right_x = self.interpolate_swing_x(right_x, right_x + stride_length, swing_progress)
                new_right_y = self.interpolate_swing_y(right_y, self.stride_width/2, swing_progress)
                new_right_z = self.interpolate_swing_z(swing_progress)
                
                # Update trajectories
                step_pattern['right_foot']['x'].append(new_right_x)
                step_pattern['right_foot']['y'].append(new_right_y)
                step_pattern['right_foot']['z'].append(new_right_z)
                
                # Left foot stays in place
                step_pattern['left_foot']['x'].append(left_x)
                step_pattern['left_foot']['y'].append(left_y)
                step_pattern['left_foot']['z'].append(0.0)
                
                # Phase label
                step_pattern['phase'].append('left_stance')
            
            # Update CoM position (follows support foot with delay)
            if step_type == 'left_swing':
                # Right foot is support, CoM moves toward it
                target_com_x = right_x + stride_length * 0.3  # CoM leads a bit
                current_com_x = step_pattern['com']['x'][-1] if step_pattern['com']['x'] else com_x
                new_com_x = current_com_x + (target_com_x - current_com_x) * (swing_progress * 0.8)
            else:
                # Left foot is support, CoM moves toward it
                target_com_x = left_x + stride_length * 0.3
                current_com_x = step_pattern['com']['x'][-1] if step_pattern['com']['x'] else com_x
                new_com_x = current_com_x + (target_com_x - current_com_x) * (swing_progress * 0.8)
            
            step_pattern['com']['x'].append(new_com_x)
            step_pattern['com']['y'].append(0.0)
            step_pattern['com']['z'].append(self.com_height)
            
            # Calculate ZMP following CoM
            step_pattern['zmp']['x'].append(new_com_x - 0.02)  # ZMP slightly behind CoM
            step_pattern['zmp']['y'].append(0.0)
            
            step_pattern['time'].append(time_point)
        
        # Phase 3: Final double support
        for t in np.arange(0, double_support_dur, self.traj_dt):
            time_point = start_time + double_support_dur + single_support_dur + t
            final_progress = t / double_support_dur
            
            # Both feet are on ground at new positions
            if step_type == 'left_swing':
                # Left foot now at new position, right foot moves forward next step
                step_pattern['left_foot']['x'].append(left_x + stride_length)
                step_pattern['left_foot']['y'].append(-self.stride_width/2)
                step_pattern['left_foot']['z'].append(0.0)
                
                step_pattern['right_foot']['x'].append(right_x)
                step_pattern['right_foot']['y'].append(right_y)
                step_pattern['right_foot']['z'].append(0.0)
            else:
                # Right foot now at new position, left foot moves forward next step
                step_pattern['left_foot']['x'].append(left_x)
                step_pattern['left_foot']['y'].append(left_y)
                step_pattern['left_foot']['z'].append(0.0)
                
                step_pattern['right_foot']['x'].append(right_x + stride_length)
                step_pattern['right_foot']['y'].append(self.stride_width/2)
                step_pattern['right_foot']['z'].append(0.0)
            
            # CoM settles at new position
            if step_type == 'left_swing':
                final_com_x = left_x + stride_length - stride_length * 0.1  # Settle behind
            else:
                final_com_x = right_x + stride_length - stride_length * 0.1
            
            step_pattern['com']['x'].append(final_com_x)
            step_pattern['com']['y'].append(0.0)
            step_pattern['com']['z'].append(self.com_height)
            
            # ZMP follows CoM
            step_pattern['zmp']['x'].append(final_com_x)
            step_pattern['zmp']['y'].append(0.0)
            
            step_pattern['phase'].append('double_support')
            step_pattern['time'].append(time_point)
        
        return step_pattern
    
    def interpolate_swing_x(self, start_x, end_x, progress):
        """
        Interpolate swing foot x position with smooth acceleration/deceleration
        """
        # Use 5th order polynomial for smooth motion
        p = progress
        return start_x + (end_x - start_x) * (10*p**3 - 15*p**4 + 6*p**5)
    
    def interpolate_swing_y(self, start_y, end_y, progress):
        """
        Interpolate swing foot y position for step width adjustment
        """
        # Direct linear interpolation for lateral movement
        return start_y + (end_y - start_y) * progress
    
    def interpolate_swing_z(self, progress):
        """
        Interpolate swing foot z position for foot clearance
        """
        # Lift foot in middle of swing phase
        if progress < self.swing_lift_ratio:
            # Lifting phase
            lift_progress = progress / self.swing_lift_ratio
            return self.step_height * (10*lift_progress**3 - 15*lift_progress**4 + 6*lift_progress**5)
        elif progress > self.swing_land_ratio:
            # Landing phase
            land_progress = (progress - self.swing_land_ratio) / (1 - self.swing_land_ratio)
            return self.step_height * (1 - (10*land_progress**3 - 15*land_progress**4 + 6*land_progress**5))
        else:
            # Peak height phase
            return self.step_height

# Example usage
pattern_gen = WalkPatternGenerator()
walk_pattern = pattern_gen.generate_walk_pattern(duration=4.0, walking_speed=0.4)

print(f"Generated walk pattern with {len(walk_pattern['time'])} time steps")
print(f"Walked distance: {walk_pattern['com']['x'][-1] - walk_pattern['com']['x'][0]:.2f} meters")
```

## Balance restoration

Maintaining and restoring balance is critical for stable bipedal locomotion. This involves detecting balance disturbances and applying corrective actions.

### Balance Disturbance Detection

```python
class BalanceDisturbanceDetector:
    def __init__(self):
        # Thresholds for disturbance detection
        self.zmp_threshold = 0.05  # meters outside support polygon
        self.com_velocity_threshold = 0.5  # m/s excessive CoM velocity
        self.angular_velocity_threshold = 0.5  # rad/s excessive body angular velocity
        self.foot_pressure_threshold = 0.1  # N minimum pressure to detect contact
        
        # Support polygon parameters
        self.foot_size_x = 0.15  # meters
        self.foot_size_y = 0.10  # meters
        
        # Filtering parameters
        self.low_pass_filter_coeff = 0.1
        self.history_length = 10  # Number of past samples to consider
        
        # Initialize history
        self.zmp_history = []
        self.com_vel_history = []
        self.ang_vel_history = []
        
    def detect_disturbance(self, robot_state):
        """
        Detect balance disturbances from robot state
        robot_state: dictionary containing sensor data
        """
        disturbances = {
            'zmp_outside_support': False,
            'excessive_com_velocity': False,
            'excessive_angular_velocity': False,
            'loss_of_support': False,
            'combined_risk': 0.0  # Overall risk score
        }
        
        # Get current state values
        zmp = robot_state.get('zmp', [0.0, 0.0])
        com_velocity = robot_state.get('com_velocity', [0.0, 0.0, 0.0])
        angular_velocity = robot_state.get('angular_velocity', [0.0, 0.0, 0.0])
        foot_positions = robot_state.get('foot_positions', {})
        foot_forces = robot_state.get('foot_forces', {})
        
        # Update history
        self.zmp_history.append(zmp.copy())
        self.com_vel_history.append(com_velocity.copy())
        self.ang_vel_history.append(angular_velocity.copy())
        
        # Keep only recent history
        if len(self.zmp_history) > self.history_length:
            self.zmp_history.pop(0)
            self.com_vel_history.pop(0)
            self.ang_vel_history.pop(0)
        
        # 1. Check if ZMP is outside support polygon
        if 'left' in foot_positions and 'right' in foot_positions:
            support_polygon = self.calculate_support_polygon(
                foot_positions['left'], 
                foot_positions['right']
            )
            disturbances['zmp_outside_support'] = not self.is_point_in_polygon(
                zmp, support_polygon
            )
        
        # 2. Check for excessive CoM velocity
        com_speed = np.linalg.norm(com_velocity[:2])  # Horizontal components
        disturbances['excessive_com_velocity'] = com_speed > self.com_velocity_threshold
        
        # 3. Check for excessive angular velocity
        ang_speed = np.linalg.norm(angular_velocity)
        disturbances['excessive_angular_velocity'] = ang_speed > self.angular_velocity_threshold
        
        # 4. Check for loss of support (no foot contact)
        left_contact = foot_forces.get('left', 0) > self.foot_pressure_threshold
        right_contact = foot_forces.get('right', 0) > self.foot_pressure_threshold
        disturbances['loss_of_support'] = not (left_contact or right_contact)
        
        # 5. Calculate combined risk score
        risk_score = 0.0
        if disturbances['zmp_outside_support']:
            risk_score += 0.4
        if disturbances['excessive_com_velocity']:
            risk_score += 0.3
        if disturbances['excessive_angular_velocity']:
            risk_score += 0.2
        if disturbances['loss_of_support']:
            risk_score += 0.1
        
        disturbances['combined_risk'] = risk_score
        
        return disturbances
    
    def calculate_support_polygon(self, left_foot_pos, right_foot_pos):
        """
        Calculate support polygon from foot positions
        """
        # Create rectangle around each foot and find convex hull
        left_points = self.foot_rectangle(left_foot_pos)
        right_points = self.foot_rectangle(right_foot_pos)
        
        # Combine all points
        all_points = left_points + right_points
        
        # Find convex hull (simplified - in practice use scipy.convex_hull)
        # For now, return all corner points
        return all_points
    
    def foot_rectangle(self, foot_pos):
        """
        Create rectangular footprint for a foot
        """
        x, y, z = foot_pos
        half_x = self.foot_size_x / 2.0
        half_y = self.foot_size_y / 2.0
        
        return [
            [x - half_x, y - half_y],
            [x + half_x, y - half_y],
            [x + half_x, y + half_y],
            [x - half_x, y + half_y]
        ]
    
    def is_point_in_polygon(self, point, polygon):
        """
        Check if a 2D point is inside a polygon using ray casting
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

# Example usage
disturbance_detector = BalanceDisturbanceDetector()

# Example robot state
robot_state = {
    'zmp': [0.2, 0.0],  # Potentially outside support
    'com_velocity': [0.6, 0.1, 0.0],  # Excessive forward velocity
    'angular_velocity': [0.2, 0.1, 0.3],  # Some angular velocity
    'foot_positions': {
        'left': [0.1, 0.15, 0.0],
        'right': [0.1, -0.15, 0.0]
    },
    'foot_forces': {
        'left': 300.0,  # Normal force (N)
        'right': 280.0
    }
}

disturbances = disturbance_detector.detect_disturbance(robot_state)
print("Detected disturbances:")
for key, value in disturbances.items():
    print(f"  {key}: {value}")
```

### Balance Restoration Controllers

```python
class BalanceRestorationController:
    def __init__(self):
        # Control parameters
        self.zmp_p_gain = 50.0  # Proportional gain for ZMP control
        self.zmp_d_gain = 10.0  # Derivative gain for ZMP control
        self.com_p_gain = 30.0  # Proportional gain for CoM control
        self.com_d_gain = 15.0  # Derivative gain for CoM control
        self.ankle_p_gain = 100.0  # Ankle control gain
        self.hip_p_gain = 80.0     # Hip control gain
        
        # Safety limits
        self.max_ankle_torque = 50.0  # Nm
        self.max_hip_torque = 100.0   # Nm
        self.max_com_adjustment = 0.05  # meters
        self.max_ankle_angle = 0.3  # radians (about 17 degrees)
        
        # Previous values for derivative terms
        self.prev_zmp_error = np.zeros(2)
        self.prev_com_error = np.zeros(3)
        self.prev_time = None
        
        # Recovery strategies
        self.recovery_strategies = {
            'ankle_strategy': 0.0,    # Ankle adjustment (stepping in place)
            'hip_strategy': 0.0,      # Hip movement strategy
            'stepping_strategy': 0.0, # Actual stepping strategy
            'arm_swing': 0.0          # Arm swing for momentum
        }
    
    def restore_balance(self, robot_state, disturbances):
        """
        Generate balance restoration commands based on detected disturbances
        """
        # Calculate time step
        current_time = time.time()
        dt = (current_time - self.prev_time) if self.prev_time else 0.01
        self.prev_time = current_time
        
        # Extract state information
        current_zmp = np.array(robot_state.get('zmp', [0.0, 0.0]))
        current_com = np.array(robot_state.get('com_position', [0.0, 0.0, 0.8]))
        current_com_vel = np.array(robot_state.get('com_velocity', [0.0, 0.0, 0.0]))
        foot_positions = robot_state.get('foot_positions', {})
        
        # Calculate desired ZMP (usually center of support polygon)
        desired_zmp = self.calculate_desired_zmp(foot_positions)
        
        # Calculate errors
        zmp_error = desired_zmp - current_zmp
        com_error = np.zeros(3)  # Will calculate based on strategy
        
        # Select appropriate recovery strategy based on disturbance severity
        strategy = self.select_recovery_strategy(disturbances, zmp_error)
        
        # Apply selected strategy
        if strategy == 'ankle':
            control_commands = self.ankle_balance_strategy(zmp_error, current_com, dt)
        elif strategy == 'hip':
            control_commands = self.hip_balance_strategy(zmp_error, current_com, current_com_vel, dt)
        elif strategy == 'stepping':
            control_commands = self.stepping_strategy(zmp_error, current_com, foot_positions)
        elif strategy == 'arm_swing':
            control_commands = self.arm_swing_strategy(zmp_error, current_com)
        else:
            control_commands = self.combined_strategy(zmp_error, current_com, current_com_vel, dt)
        
        # Apply safety limits
        control_commands = self.apply_safety_limits(control_commands)
        
        return control_commands, strategy
    
    def calculate_desired_zmp(self, foot_positions):
        """
        Calculate desired ZMP based on foot positions
        """
        if len(foot_positions) == 0:
            return np.array([0.0, 0.0])
        
        if len(foot_positions) == 1:
            # Single support - ZMP near stance foot
            foot_pos = list(foot_positions.values())[0]
            return np.array([foot_pos[0], foot_pos[1]])
        else:
            # Double support - ZMP between feet
            positions = list(foot_positions.values())
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            return np.array([avg_x, avg_y])
    
    def select_recovery_strategy(self, disturbances, zmp_error):
        """
        Select appropriate balance recovery strategy based on disturbance type and severity
        """
        zmp_distance = np.linalg.norm(zmp_error)
        
        # Classify disturbance severity
        if disturbances['combined_risk'] > 0.7:
            # Severe disturbance - stepping may be necessary
            if zmp_distance > 0.1:  # Well outside support
                return 'stepping'
            else:
                return 'combined'
        elif disturbances['combined_risk'] > 0.4:
            # Moderate disturbance
            if zmp_distance > 0.05:  # Outside normal range
                return 'hip'
            else:
                return 'ankle'
        else:
            # Mild disturbance - use fine adjustments
            return 'ankle'
    
    def ankle_balance_strategy(self, zmp_error, current_com, dt):
        """
        Ankle-based balance strategy for small disturbances
        """
        # Calculate ankle torques to move ZMP back to desired position
        ankle_roll_torque = self.ankle_p_gain * zmp_error[1] * 0.5  # Y-direction affects roll
        ankle_pitch_torque = self.ankle_p_gain * zmp_error[0] * 0.5  # X-direction affects pitch
        
        # Apply limits
        ankle_roll_torque = np.clip(ankle_roll_torque, -self.max_ankle_torque, self.max_ankle_torque)
        ankle_pitch_torque = np.clip(ankle_pitch_torque, -self.max_ankle_torque, self.max_ankle_torque)
        
        # Generate control commands
        commands = {
            'left_ankle_roll': ankle_roll_torque,
            'left_ankle_pitch': ankle_pitch_torque,
            'right_ankle_roll': -ankle_roll_torque,  # Opposite for balance
            'right_ankle_pitch': ankle_pitch_torque,
            'coordinated_body_motion': False
        }
        
        return commands
    
    def hip_balance_strategy(self, zmp_error, current_com, current_com_vel, dt):
        """
        Hip-based balance strategy for moderate disturbances
        """
        # Calculate CoM adjustment needed
        com_adjustment_x = -zmp_error[0] * 0.3  # Map ZMP error to CoM adjustment
        com_adjustment_y = -zmp_error[1] * 0.3
        
        # Limit adjustment
        com_adjustment_x = np.clip(com_adjustment_x, -self.max_com_adjustment, self.max_com_adjustment)
        com_adjustment_y = np.clip(com_adjustment_y, -self.max_com_adjustment, self.max_com_adjustment)
        
        # Calculate required hip torques
        hip_roll_torque = self.hip_p_gain * com_adjustment_y
        hip_pitch_torque = self.hip_p_gain * com_adjustment_x
        
        # Apply limits
        hip_roll_torque = np.clip(hip_roll_torque, -self.max_hip_torque, self.max_hip_torque)
        hip_pitch_torque = np.clip(hip_pitch_torque, -self.max_hip_torque, self.max_hip_torque)
        
        commands = {
            'left_hip_roll': hip_roll_torque,
            'left_hip_pitch': hip_pitch_torque,
            'right_hip_roll': -hip_roll_torque,
            'right_hip_pitch': hip_pitch_torque,
            'coordinated_body_motion': True
        }
        
        return commands
    
    def stepping_strategy(self, zmp_error, current_com, foot_positions):
        """
        Stepping strategy for severe disturbances
        """
        # Determine where to step based on ZMP position
        # Step in the direction opposite to ZMP displacement
        step_direction = -zmp_error
        step_direction = step_direction / np.linalg.norm(step_direction) if np.linalg.norm(step_direction) > 0.01 else np.array([1, 0])
        
        # Calculate step location
        current_support_foot = self.get_current_support_foot(foot_positions)
        step_position = self.calculate_step_position(current_support_foot, step_direction)
        
        commands = {
            'step_target': step_position,
            'initiate_stepping_sequence': True,
            'swing_leg_control': self.calculate_swing_leg_control(step_position),
            'stance_leg_adjustment': self.calculate_stance_leg_adjustment(),
            'weight_transfer_timing': self.calculate_weight_transfer_timing()
        }
        
        return commands
    
    def arm_swing_strategy(self, zmp_error, current_com):
        """
        Arm swing strategy to create corrective angular momentum
        """
        # Calculate required arm motion to generate corrective momentum
        arm_torque_x = -zmp_error[1] * 20  # Swing arms to create corrective moment
        arm_torque_y = zmp_error[0] * 20
        
        commands = {
            'left_shoulder_roll': -arm_torque_y,
            'left_shoulder_pitch': arm_torque_x,
            'right_shoulder_roll': arm_torque_y,
            'right_shoulder_pitch': arm_torque_x,
            'arm_swing_momentum': True
        }
        
        return commands
    
    def combined_strategy(self, zmp_error, current_com, current_com_vel, dt):
        """
        Combined strategy using multiple approaches
        """
        # Use a weighted combination of different strategies
        ankle_cmd = self.ankle_balance_strategy(zmp_error * 0.5, current_com, dt)
        hip_cmd = self.hip_balance_strategy(zmp_error * 0.3, current_com, current_com_vel, dt)
        arm_cmd = self.arm_swing_strategy(zmp_error * 0.2, current_com)
        
        # Combine commands (simplified - in practice need to coordinate properly)
        combined_cmd = {
            'ankle_roll': 0.6 * ankle_cmd.get('left_ankle_roll', 0) + 0.4 * hip_cmd.get('left_hip_roll', 0),
            'ankle_pitch': 0.6 * ankle_cmd.get('left_ankle_pitch', 0) + 0.4 * hip_cmd.get('left_hip_pitch', 0),
            'hip_roll': 0.4 * hip_cmd.get('left_hip_roll', 0),
            'hip_pitch': 0.4 * hip_cmd.get('left_hip_pitch', 0),
            'arm_adjustment': arm_cmd.get('left_shoulder_roll', 0),
            'coordinated_control': True
        }
        
        return combined_cmd
    
    def get_current_support_foot(self, foot_positions):
        """
        Determine which foot is currently supporting the weight
        """
        # Simplified - in practice use force sensors
        # For now, return the foot closest to CoM projection
        return 'left' if 'left' in foot_positions else 'right'
    
    def calculate_step_position(self, support_foot, direction):
        """
        Calculate appropriate step position
        """
        # Simplified calculation - in practice consider dynamic constraints
        step_distance = 0.3  # meters
        step_pos = np.array([0.3, 0.0])  # Default forward step
        step_pos = step_pos + direction * step_distance * 0.5  # Adjust based on direction
        
        return step_pos.tolist()
    
    def calculate_swing_leg_control(self, target_position):
        """
        Calculate control for swing leg to reach target position
        """
        # This would involve inverse kinematics and trajectory planning
        return {
            'trajectory': 'minimum_jerk_to_target',
            'timing': 'smooth_transition',
            'foot_placement': 'controlled_landing'
        }
    
    def calculate_stance_leg_adjustment(self):
        """
        Adjust stance leg during stepping
        """
        return {
            'weight_shift': 'controlled',
            'knee_stiffness': 'increased',
            'ankle_adaptation': 'active'
        }
    
    def calculate_weight_transfer_timing(self):
        """
        Calculate optimal timing for weight transfer during step
        """
        return {
            'double_support_duration': 0.1,  # seconds
            'transfer_speed': 'moderate',
            'stability_checks': True
        }
    
    def apply_safety_limits(self, commands):
        """
        Apply safety limits to control commands
        """
        # Implement safety limits for all commands
        limited_commands = commands.copy()
        
        # Example: limit joint torques
        for key, value in limited_commands.items():
            if 'torque' in key:
                limited_commands[key] = np.clip(value, -100.0, 100.0)  # Example limits
        
        return limited_commands

# Example usage
restoration_controller = BalanceRestorationController()

# Example robot state with disturbances
disturbed_state = {
    'zmp': [0.15, 0.05],  # Outside normal support
    'com_position': [0.1, 0.02, 0.78],
    'com_velocity': [0.4, 0.05, 0.0],
    'foot_positions': {
        'left': [0.0, 0.15, 0.0],
        'right': [0.0, -0.15, 0.0]
    }
}

disturbances_example = {
    'zmp_outside_support': True,
    'excessive_com_velocity': True,
    'excessive_angular_velocity': False,
    'loss_of_support': False,
    'combined_risk': 0.7
}

commands, strategy = restoration_controller.restore_balance(disturbed_state, disturbances_example)
print(f"Applied balance restoration strategy: {strategy}")
print(f"Commands: {commands}")
```

## Leg trajectory planning

Planning appropriate leg trajectories is essential for stable and efficient bipedal locomotion. This involves coordinating the movement of the swing leg while maintaining balance.

### Swing Leg Trajectory Planning

```python
class SwingLegTrajectoryPlanner:
    def __init__(self):
        # Trajectory parameters
        self.foot_clearance = 0.10  # meters of foot clearance
        self.foot_approach_distance = 0.05  # meters before touchdown
        self.swing_duration_ratio = 0.7  # 70% of step time for swing
        self.touchdown_duration_ratio = 0.1  # 10% for touchdown
        
        # Polynomial trajectory parameters
        self.min_trajectory_points = 10
        self.max_trajectory_points = 50
        
        # Gait-specific parameters
        self.walk_style = 'natural'  # 'natural', 'cautious', 'dynamic'
        self.step_height_multiplier = 1.0
        self.landing_smoothness = 0.5  # 0=sharp, 1=very smooth
    
    def plan_swing_trajectory(self, start_pos, target_pos, step_duration, gait_params=None):
        """
        Plan swing leg trajectory from start to target position
        """
        if gait_params is None:
            gait_params = {
                'speed': 0.3,  # m/s
                'step_length': 0.3,  # m
                'step_width': 0.25,  # m
                'style': self.walk_style
            }
        
        # Adjust parameters based on gait style
        if gait_params['style'] == 'cautious':
            self.step_height_multiplier = 1.2
            self.landing_smoothness = 0.8
        elif gait_params['style'] == 'dynamic':
            self.step_height_multiplier = 0.8
            self.landing_smoothness = 0.3
        else:  # natural
            self.step_height_multiplier = 1.0
            self.landing_smoothness = 0.5
        
        # Calculate trajectory timing
        swing_duration = step_duration * self.swing_duration_ratio
        touchdown_duration = step_duration * self.touchdown_duration_ratio
        liftoff_duration = step_duration - swing_duration - touchdown_duration
        
        # Calculate trajectory points
        num_points = max(self.min_trajectory_points, 
                        min(self.max_trajectory_points, int(swing_duration / 0.01)))
        dt = swing_duration / num_points if num_points > 0 else 0.01
        
        trajectory = {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': [],
            'phase': []  # 'liftoff', 'swing', 'touchdown'
        }
        
        # Phase 1: Liftoff
        liftoff_points = int(liftoff_duration / dt) if dt > 0 else 0
        liftoff_trajectory = self.plan_liftoff_trajectory(start_pos, dt, liftoff_points)
        
        # Add liftoff to trajectory
        for i in range(len(liftoff_trajectory['position'])):
            trajectory['time'].append(i * dt)
            trajectory['position'].append(liftoff_trajectory['position'][i])
            trajectory['velocity'].append(liftoff_trajectory['velocity'][i])
            trajectory['acceleration'].append(liftoff_trajectory['acceleration'][i])
            trajectory['phase'].append('liftoff')
        
        # Phase 2: Swing
        swing_trajectory = self.plan_swing_trajectory_core(
            liftoff_trajectory['position'][-1] if liftoff_trajectory['position'] else start_pos,
            target_pos,
            dt, num_points
        )
        
        # Add swing to trajectory
        start_time = trajectory['time'][-1] if trajectory['time'] else 0
        for i in range(len(swing_trajectory['position'])):
            trajectory['time'].append(start_time + i * dt)
            trajectory['position'].append(swing_trajectory['position'][i])
            trajectory['velocity'].append(swing_trajectory['velocity'][i])
            trajectory['acceleration'].append(swing_trajectory['acceleration'][i])
            trajectory['phase'].append('swing')
        
        # Phase 3: Touchdown
        touchdown_points = int(touchdown_duration / dt) if dt > 0 else 5
        touchdown_trajectory = self.plan_touchdown_trajectory(
            swing_trajectory['position'][-1] if swing_trajectory['position'] else target_pos,
            target_pos, dt, touchdown_points
        )
        
        # Add touchdown to trajectory
        start_time = trajectory['time'][-1] if trajectory['time'] else 0
        for i in range(len(touchdown_trajectory['position'])):
            trajectory['time'].append(start_time + i * dt)
            trajectory['position'].append(touchdown_trajectory['position'][i])
            trajectory['velocity'].append(touchdown_trajectory['velocity'][i])
            trajectory['acceleration'].append(touchdown_trajectory['acceleration'][i])
            trajectory['phase'].append('touchdown')
        
        return trajectory
    
    def plan_liftoff_trajectory(self, start_pos, dt, num_points):
        """
        Plan liftoff trajectory (foot lifting from ground)
        """
        trajectory = {'position': [], 'velocity': [], 'acceleration': []}
        
        if num_points <= 0:
            return trajectory
        
        # Start from ground contact position
        start_pos_lifted = start_pos.copy()
        start_pos_lifted[2] += 0.01  # Slightly above ground
        
        # End at full clearance height
        end_pos = start_pos.copy()
        end_pos[2] = start_pos[2] + self.foot_clearance * self.step_height_multiplier
        
        for i in range(num_points):
            progress = i / (num_points - 1) if num_points > 1 else 0
            
            # Use 5th order polynomial for smooth liftoff
            p = progress
            height_factor = (10*p**3 - 15*p**4 + 6*p**5)
            
            pos = start_pos_lifted.copy()
            pos[2] = start_pos[2] + (end_pos[2] - start_pos[2]) * height_factor
            
            # Calculate velocity and acceleration
            if i > 0:
                vel = [(pos[j] - trajectory['position'][-1][j]) / dt for j in range(3)]
            else:
                vel = [0, 0, 0]  # Start with zero velocity
            
            trajectory['position'].append(pos)
            trajectory['velocity'].append(vel)
            trajectory['acceleration'].append([0, 0, 0])  # Simplified
        
        return trajectory
    
    def plan_swing_trajectory_core(self, start_pos, end_pos, dt, num_points):
        """
        Plan main swing trajectory (parabolic arc)
        """
        trajectory = {'position': [], 'velocity': [], 'acceleration': []}
        
        # Calculate intermediate waypoints
        mid_x = (start_pos[0] + end_pos[0]) / 2.0
        mid_y = (start_pos[1] + end_pos[1]) / 2.0
        mid_z = max(start_pos[2], end_pos[2]) + self.foot_clearance * self.step_height_multiplier
        
        # Use parabolic path
        for i in range(num_points):
            progress = i / (num_points - 1) if num_points > 1 else 0
            
            # X and Y follow linear interpolation
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            
            # Z follows parabolic path
            if progress < 0.5:
                # Ascending part
                asc_progress = progress * 2  # Scale to 0-1
                z = start_pos[2] + (mid_z - start_pos[2]) * (10*asc_progress**3 - 15*asc_progress**4 + 6*asc_progress**5)
            else:
                # Descending part
                desc_progress = (progress - 0.5) * 2  # Scale to 0-1
                z = mid_z + (end_pos[2] - mid_z) * (10*desc_progress**3 - 15*desc_progress**4 + 6*desc_progress**5)
            
            pos = [x, y, z]
            
            # Calculate velocity
            if i > 0:
                vel = [(pos[j] - trajectory['position'][-1][j]) / dt for j in range(3)]
            else:
                vel = [0, 0, 0]
            
            trajectory['position'].append(pos)
            trajectory['velocity'].append(vel)
            trajectory['acceleration'].append([0, 0, 0])  # Simplified
        
        return trajectory
    
    def plan_touchdown_trajectory(self, start_pos, end_pos, dt, num_points):
        """
        Plan touchdown trajectory (smooth landing)
        """
        trajectory = {'position': [], 'velocity': [], 'acceleration': []}
        
        for i in range(num_points):
            progress = i / (num_points - 1) if num_points > 1 else 0
            
            # Use smooth landing profile based on landing_smoothness
            smooth_factor = self.landing_smoothness
            landing_curve = (10*progress**3 - 15*progress**4 + 6*progress**5) * smooth_factor + \
                           progress * (1 - smooth_factor)  # Blend smooth and linear
            
            pos = [
                start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                start_pos[1] + (end_pos[1] - start_pos[1]) * progress,
                start_pos[2] + (end_pos[2] - start_pos[2]) * landing_curve
            ]
            
            # Calculate velocity (aim for zero at touchdown)
            if i > 0:
                vel = [(pos[j] - trajectory['position'][-1][j]) / dt for j in range(3)]
            else:
                vel = [(end_pos[j] - start_pos[j]) / (num_points * dt) for j in range(3)]
            
            trajectory['position'].append(pos)
            trajectory['velocity'].append(vel)
            trajectory['acceleration'].append([0, 0, 0])  # Simplified
        
        return trajectory
    
    def optimize_trajectory(self, trajectory, robot_constraints):
        """
        Optimize trajectory considering robot constraints
        """
        # Apply joint limits, velocity limits, and acceleration limits
        optimized_trajectory = trajectory.copy()
        
        # Example optimization: limit velocities
        max_vel = robot_constraints.get('max_joint_velocity', 2.0)  # rad/s
        max_acc = robot_constraints.get('max_joint_acceleration', 5.0)  # rad/s^2
        
        for i in range(len(optimized_trajectory['velocity'])):
            # Limit velocity magnitude
            vel_mag = np.linalg.norm(optimized_trajectory['velocity'][i])
            if vel_mag > max_vel:
                scale = max_vel / vel_mag
                optimized_trajectory['velocity'][i] = [
                    v * scale for v in optimized_trajectory['velocity'][i]
                ]
        
        return optimized_trajectory
    
    def generate_ankle_trajectory(self, foot_trajectory, foot_orientation_start, foot_orientation_end):
        """
        Generate corresponding ankle joint trajectories to maintain foot orientation
        """
        # This would involve inverse kinematics to determine required ankle joint angles
        # to maintain the desired foot trajectory and orientation
        
        ankle_trajectory = {
            'roll': [],
            'pitch': [],
            'yaw': []
        }
        
        # Simplified example - in practice this requires full IK solution
        for foot_pos in foot_trajectory['position']:
            # Calculate required ankle angles based on foot position and desired orientation
            # This is highly simplified
            ankle_roll = 0.0  # Would be calculated based on terrain and balance needs
            ankle_pitch = 0.0  # Would be calculated based on forward motion needs
            ankle_yaw = 0.0   # Would be calculated based on turning needs
            
            ankle_trajectory['roll'].append(ankle_roll)
            ankle_trajectory['pitch'].append(ankle_pitch)
            ankle_trajectory['yaw'].append(ankle_yaw)
        
        return ankle_trajectory

# Example usage
swing_planner = SwingLegTrajectoryPlanner()

# Define a step from one position to another
start_pos = [0.0, 0.15, 0.0]  # Left foot position
target_pos = [0.3, -0.15, 0.0]  # Target position for next step
step_duration = 0.8  # seconds

gait_params = {
    'speed': 0.4,
    'step_length': 0.3,
    'step_width': 0.3,
    'style': 'natural'
}

trajectory = swing_planner.plan_swing_trajectory(start_pos, target_pos, step_duration, gait_params)

print(f"Planned swing trajectory with {len(trajectory['position'])} points")
print(f"Trajectory duration: {trajectory['time'][-1]:.2f} seconds")
print(f"Start position: {trajectory['position'][0]}")
print(f"End position: {trajectory['position'][-1]}")

# Apply robot constraints
robot_constraints = {
    'max_joint_velocity': 1.5,
    'max_joint_acceleration': 3.0
}

optimized_trajectory = swing_planner.optimize_trajectory(trajectory, robot_constraints)
print("Trajectory optimized for robot constraints")
```

### Gait Phase Synchronization

Proper synchronization between the two legs is crucial for stable walking:

```python
class GaitSynchronizer:
    def __init__(self):
        # Phase relationship parameters
        self.phase_offset = np.pi  # 180 degrees between legs
        self.phase_sync_bandwidth = 0.1  # Tolerance for phase synchronization
        self.step_timing_precision = 0.05  # seconds precision for step timing
        
        # Double support phase parameters
        self.min_double_support = 0.05  # minimum double support time
        self.max_double_support = 0.2   # maximum double support time
        
        # Synchronization control parameters
        self.sync_p_gain = 10.0  # Proportional gain for phase synchronization
        self.sync_d_gain = 5.0  # Derivative gain for phase synchronization
    
    def synchronize_gait_phases(self, left_leg_state, right_leg_state, gait_params):
        """
        Synchronize gait phases between left and right legs
        """
        # Extract phase information from leg states
        left_phase = left_leg_state.get('phase', 0.0)
        right_phase = right_leg_state.get('phase', 0.0)
        
        # Calculate phase error (should be approximately π apart)
        desired_phase_diff = self.phase_offset
        actual_phase_diff = self.calculate_phase_difference(left_phase, right_phase)
        phase_error = desired_phase_diff - actual_phase_diff
        
        # Adjust gait timing based on phase error
        sync_adjustment = self.calculate_sync_adjustment(phase_error, gait_params)
        
        synchronization_info = {
            'phase_error': phase_error,
            'sync_adjustment': sync_adjustment,
            'recommended_timing': self.adjust_step_timing(
                left_leg_state, right_leg_state, sync_adjustment
            ),
            'stability_metrics': self.calculate_stability_metrics(
                left_leg_state, right_leg_state
            )
        }
        
        return synchronization_info
    
    def calculate_phase_difference(self, phase1, phase2):
        """
        Calculate phase difference accounting for circular nature of phase
        """
        diff = phase1 - phase2
        # Normalize to [-π, π] range
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return abs(diff)
    
    def calculate_sync_adjustment(self, phase_error, gait_params):
        """
        Calculate adjustment needed to synchronize phases
        """
        # Use PD controller for phase synchronization
        adjustment = self.sync_p_gain * phase_error
        
        # Limit adjustment based on gait parameters
        max_adjustment = gait_params.get('step_duration', 0.8) * 0.1  # 10% of step time
        adjustment = np.clip(adjustment, -max_adjustment, max_adjustment)
        
        return adjustment
    
    def adjust_step_timing(self, left_state, right_state, sync_adjustment):
        """
        Adjust step timing based on synchronization needs
        """
        # Calculate new timing parameters
        current_left_duration = left_state.get('step_duration', 0.8)
        current_right_duration = right_state.get('step_duration', 0.8)
        
        # Apply adjustment to the lagging leg
        if left_state.get('phase', 0) < right_state.get('phase', 0):
            # Left leg is lagging, extend its step
            new_left_duration = current_left_duration + sync_adjustment
            new_right_duration = current_right_duration  # Keep right unchanged
        else:
            # Right leg is lagging, extend its step
            new_right_duration = current_right_duration + sync_adjustment
            new_left_duration = current_left_duration  # Keep left unchanged
        
        return {
            'left_step_duration': new_left_duration,
            'right_step_duration': new_right_duration,
            'timing_changed': abs(sync_adjustment) > 0.01
        }
    
    def calculate_stability_metrics(self, left_state, right_state):
        """
        Calculate metrics for gait stability
        """
        metrics = {}
        
        # Calculate double support duration
        left_ds = left_state.get('double_support_duration', 0.08)
        right_ds = right_state.get('double_support_duration', 0.08)
        avg_double_support = (left_ds + right_ds) / 2.0
        
        # Calculate step symmetry
        left_step_len = left_state.get('step_length', 0.3)
        right_step_len = right_state.get('step_length', 0.3)
        step_symmetry = 1.0 - abs(left_step_len - right_step_len) / max(left_step_len, right_step_len, 0.01)
        
        # Calculate phase coordination
        phase_coordination = 1.0 - abs(self.calculate_phase_difference(
            left_state.get('phase', 0), 
            right_state.get('phase', np.pi)
        )) / np.pi
        
        metrics = {
            'double_support_duration': avg_double_support,
            'step_symmetry': step_symmetry,
            'phase_coordination': phase_coordination,
            'overall_stability_score': (avg_double_support * 0.2 + step_symmetry * 0.4 + phase_coordination * 0.4)
        }
        
        return metrics
    
    def generate_entrainment_signal(self, current_phase, target_phase, strength=1.0):
        """
        Generate entrainment signal to help synchronize gait
        """
        phase_diff = self.calculate_phase_difference(current_phase, target_phase)
        
        # Create sinusoidal entrainment signal
        entrainment = strength * np.sin(phase_diff)
        
        return entrainment

# Example usage
synchronizer = GaitSynchronizer()

# Example leg states
left_leg_state = {
    'phase': 0.1,  # Early in stance phase
    'step_duration': 0.8,
    'step_length': 0.31,
    'double_support_duration': 0.07
}

right_leg_state = {
    'phase': 3.0,  # Later in swing phase (around π)
    'step_duration': 0.78,
    'step_length': 0.29,
    'double_support_duration': 0.09
}

gait_params = {
    'step_duration': 0.8,
    'walking_speed': 0.4
}

sync_info = synchronizer.synchronize_gait_phases(left_leg_state, right_leg_state, gait_params)

print("Gait Synchronization Results:")
print(f"Phase Error: {sync_info['phase_error']:.3f}")
print(f"Sync Adjustment: {sync_info['sync_adjustment']:.3f}")
print(f"Recommended Timing: {sync_info['recommended_timing']}")
print(f"Stability Metrics: {sync_info['stability_metrics']}")
```

## Fall prevention & recovery

Fall prevention and recovery mechanisms are critical for safe bipedal locomotion. These systems detect potential falls and initiate protective responses.

### Fall Detection System

```python
class FallDetectionSystem:
    def __init__(self):
        # Critical thresholds for fall detection
        self.angle_threshold = 30.0 * np.pi / 180.0  # 30 degrees in radians
        self.angular_velocity_threshold = 2.0  # rad/s
        self.com_height_threshold = 0.3  # meters (if CoM drops below this)
        self.zmp_escape_threshold = 0.2  # meters outside support polygon
        self.velocity_threshold = 1.0  # m/s excessive CoM velocity
        
        # Prediction horizon
        self.prediction_horizon = 0.5  # seconds to predict fall
        self.prediction_dt = 0.01  # time step for prediction
        
        # Historical data for trend analysis
        self.history_length = 50  # number of samples to keep
        self.angle_history = []
        self.omega_history = []
        self.com_height_history = []
        self.zmp_distance_history = []
        
        # Weighting factors for different fall indicators
        self.weights = {
            'angle': 0.3,
            'angular_velocity': 0.2,
            'com_height': 0.2,
            'zmp_escape': 0.2,
            'velocity': 0.1
        }
    
    def detect_fall_risk(self, robot_state):
        """
        Detect risk of falling based on robot state
        """
        # Extract state information
        imu_data = robot_state.get('imu', {})
        attitude = imu_data.get('attitude', [0, 0, 0, 1])  # w, x, y, z quaternion
        angular_velocity = imu_data.get('angular_velocity', [0, 0, 0])
        com_position = robot_state.get('com_position', [0, 0, 0.8])
        zmp_position = robot_state.get('zmp', [0, 0])
        foot_positions = robot_state.get('foot_positions', {})
        com_velocity = robot_state.get('com_velocity', [0, 0, 0])
        
        # Calculate angles from quaternion
        roll, pitch, yaw = self.quaternion_to_euler(attitude)
        total_angle = np.sqrt(roll**2 + pitch**2)  # Combined tilt angle
        
        # Calculate support polygon and ZMP distance
        if foot_positions:
            support_polygon = self.calculate_support_polygon(foot_positions)
            zmp_distance = self.distance_to_polygon(zmp_position, support_polygon)
        else:
            zmp_distance = np.inf  # No support = high fall risk
        
        # Calculate CoM height
        com_height = com_position[2]
        
        # Calculate CoM velocity magnitude
        com_speed = np.linalg.norm(com_velocity)
        
        # Update historical data
        self.angle_history.append(total_angle)
        self.omega_history.append(np.linalg.norm(angular_velocity))
        self.com_height_history.append(com_height)
        self.zmp_distance_history.append(zmp_distance)
        
        # Keep only recent history
        if len(self.angle_history) > self.history_length:
            self.angle_history.pop(0)
            self.omega_history.pop(0)
            self.com_height_history.pop(0)
            self.zmp_distance_history.pop(0)
        
        # Calculate risk factors
        angle_risk = self.sigmoid(total_angle, self.angle_threshold, 1.0)
        omega_risk = self.sigmoid(np.linalg.norm(angular_velocity), self.angular_velocity_threshold, 1.0)
        height_risk = self.sigmoid(self.com_height_threshold, com_height, -1.0)  # Lower is riskier
        zmp_risk = self.sigmoid(zmp_distance, self.zmp_escape_threshold, 1.0)
        velocity_risk = self.sigmoid(com_speed, self.velocity_threshold, 1.0)
        
        # Calculate weighted fall risk score
        risk_score = (
            self.weights['angle'] * angle_risk +
            self.weights['angular_velocity'] * omega_risk +
            self.weights['com_height'] * height_risk +
            self.weights['zmp_escape'] * zmp_risk +
            self.weights['velocity'] * velocity_risk
        )
        
        # Predict future state
        predicted_risk = self.predict_fall_risk(
            total_angle, np.linalg.norm(angular_velocity), 
            com_height, zmp_distance, com_speed
        )
        
        # Determine fall risk level
        if risk_score > 0.8 or predicted_risk > 0.9:
            risk_level = 'IMMINENT_FALL'
            confidence = risk_score
        elif risk_score > 0.5 or predicted_risk > 0.7:
            risk_level = 'HIGH_FALL_RISK'
            confidence = risk_score
        elif risk_score > 0.3 or predicted_risk > 0.5:
            risk_level = 'MODERATE_FALL_RISK'
            confidence = risk_score
        else:
            risk_level = 'LOW_FALL_RISK'
            confidence = risk_score
        
        fall_detection_result = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'predicted_risk': predicted_risk,
            'confidence': confidence,
            'risk_factors': {
                'angle_risk': angle_risk,
                'omega_risk': omega_risk,
                'height_risk': height_risk,
                'zmp_risk': zmp_risk,
                'velocity_risk': velocity_risk
            },
            'current_state': {
                'total_angle': total_angle,
                'angular_velocity': np.linalg.norm(angular_velocity),
                'com_height': com_height,
                'zmp_distance': zmp_distance,
                'com_speed': com_speed
            }
        }
        
        return fall_detection_result
    
    def sigmoid(self, x, threshold, direction=1.0):
        """
        Sigmoid function for smooth thresholding
        direction: 1.0 for higher values increasing risk, -1.0 for lower values increasing risk
        """
        if direction > 0:
            # Higher x increases risk
            return 1.0 / (1.0 + np.exp(-(x - threshold) * 5))
        else:
            # Lower x increases risk
            return 1.0 / (1.0 + np.exp((x - threshold) * 5))
    
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        """
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.pi / 2 if sinp > 0 else -np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def calculate_support_polygon(self, foot_positions):
        """
        Calculate support polygon from foot positions
        """
        if len(foot_positions) == 0:
            return []
        
        if len(foot_positions) == 1:
            # Single foot support - small polygon around foot
            foot_pos = list(foot_positions.values())[0]
            x, y, z = foot_pos
            return [[x-0.05, y-0.05], [x+0.05, y-0.05], [x+0.05, y+0.05], [x-0.05, y+0.05]]
        
        # Two feet - create polygon encompassing both
        points = []
        for pos in foot_positions.values():
            x, y, z = pos
            # Add rectangle around each foot
            points.extend([
                [x - 0.075, y - 0.05],
                [x + 0.075, y - 0.05],
                [x + 0.075, y + 0.05],
                [x - 0.075, y + 0.05]
            ])
        
        # Calculate convex hull (simplified)
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return [points[i] for i in hull.vertices]
        except:
            # If calculation fails, return a simple rectangle
            return points[:4]
    
    def distance_to_polygon(self, point, polygon):
        """
        Calculate minimum distance from point to polygon
        """
        if not polygon:
            return np.inf
        
        # Calculate distance to each edge
        min_dist = np.inf
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            dist = self.distance_point_to_line_segment(point, polygon[i], polygon[j])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def distance_point_to_line_segment(self, point, line_start, line_end):
        """
        Calculate distance from point to line segment
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # Squared length of the line segment
        length_sq = dx*dx + dy*dy
        
        if length_sq == 0:
            # Line segment is actually a point
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate projection of point onto line
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        
        # Calculate closest point on line segment
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        # Distance from point to projected point
        return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
    
    def predict_fall_risk(self, current_angle, current_omega, current_height, 
                         current_zmp_dist, current_velocity):
        """
        Predict fall risk in the near future based on current trends
        """
        # Simple prediction using current rates of change
        # In practice, this would use more sophisticated models
        
        # Predict angle in the future
        predicted_angle = current_angle + current_omega * self.prediction_horizon
        
        # Predict CoM height (assuming it's dropping)
        if len(self.com_height_history) > 1:
            height_rate = (self.com_height_history[-1] - self.com_height_history[0]) / len(self.com_height_history) * 100
            predicted_height = current_height + height_rate * self.prediction_horizon
        else:
            predicted_height = current_height
        
        # Predict ZMP distance (getting worse if current trend continues)
        if len(self.zmp_distance_history) > 1:
            zmp_rate = (self.zmp_distance_history[-1] - self.zmp_distance_history[0]) / len(self.zmp_distance_history) * 100
            predicted_zmp_dist = current_zmp_dist + zmp_rate * self.prediction_horizon
        else:
            predicted_zmp_dist = current_zmp_dist
        
        # Calculate predicted risk factors
        pred_angle_risk = self.sigmoid(predicted_angle, self.angle_threshold, 1.0)
        pred_height_risk = self.sigmoid(self.com_height_threshold, predicted_height, -1.0)
        pred_zmp_risk = self.sigmoid(predicted_zmp_dist, self.zmp_escape_threshold, 1.0)
        
        # Weighted prediction
        predicted_risk = (
            self.weights['angle'] * pred_angle_risk +
            self.weights['com_height'] * pred_height_risk +
            self.weights['zmp_escape'] * pred_zmp_risk
        )
        
        return predicted_risk

# Example usage
fall_detector = FallDetectionSystem()

# Example robot state with potential fall conditions
robot_state_example = {
    'imu': {
        'attitude': [0.9, 0.1, 0.2, 0.3],  # Somewhat tilted
        'angular_velocity': [0.8, 1.2, 0.1]  # High angular velocity
    },
    'com_position': [0.05, 0.02, 0.5],  # Low CoM height
    'zmp': [0.15, 0.08],  # Outside normal range
    'foot_positions': {
        'left': [0.0, 0.15, 0.0],
        'right': [0.1, -0.15, 0.0]
    },
    'com_velocity': [0.8, 0.2, 0.1]  # High velocity
}

fall_result = fall_detector.detect_fall_risk(robot_state_example)
print("Fall Detection Result:")
print(f"Risk Level: {fall_result['risk_level']}")
print(f"Risk Score: {fall_result['risk_score']:.3f}")
print(f"Predicted Risk: {fall_result['predicted_risk']:.3f}")
print(f"Risk Factors: {fall_result['risk_factors']}")
```

### Fall Recovery Strategies

```python
class FallRecoverySystem:
    def __init__(self):
        # Recovery strategy parameters
        self.critical_angle_threshold = 45.0 * np.pi / 180.0  # 45 degrees
        self.recovery_time_limit = 0.3  # seconds to initiate recovery
        self.impact_absorption_time = 0.2  # seconds for impact absorption
        
        # Available recovery strategies
        self.strategies = {
            'arm_swing': {'priority': 1, 'activation_angle': 20.0*np.pi/180.0, 'effectiveness': 0.7},
            'hip_strategy': {'priority': 2, 'activation_angle': 25.0*np.pi/180.0, 'effectiveness': 0.6},
            'stepping': {'priority': 3, 'activation_angle': 30.0*np.pi/180.0, 'effectiveness': 0.8},
            'parachute': {'priority': 4, 'activation_angle': 40.0*np.pi/180.0, 'effectiveness': 0.9},  # Simulated
            'crouch': {'priority': 5, 'activation_angle': 35.0*np.pi/180.0, 'effectiveness': 0.5}
        }
        
        # Strategy execution parameters
        self.arm_swing_torque = 30.0  # Nm
        self.hip_torque_limit = 80.0  # Nm
        self.leg_swing_torque = 50.0  # Nm
        
        # Impact mitigation parameters
        self.impact_absorption_joints = ['knees', 'hips', 'ankles']
        self.impact_absorption_stiffness = 0.3  # Reduced stiffness during impact
        self.impact_absorption_damping = 2.0   # Increased damping during impact
    
    def select_recovery_strategy(self, fall_detection_result):
        """
        Select appropriate fall recovery strategy based on fall detection results
        """
        current_angle = fall_detection_result['current_state']['total_angle']
        predicted_risk = fall_detection_result['predicted_risk']
        risk_level = fall_detection_result['risk_level']
        
        # Sort strategies by priority
        available_strategies = []
        for name, params in self.strategies.items():
            if current_angle >= params['activation_angle']:
                available_strategies.append((params['priority'], name, params))
        
        available_strategies.sort()  # Sort by priority (lower number = higher priority)
        
        # Select strategy based on risk level and available options
        if risk_level == 'IMMINENT_FALL':
            # Use highest priority available strategy
            if available_strategies:
                return available_strategies[0][1]  # Return strategy name
            else:
                # No active strategy available, default to crouching
                return 'crouch'
        elif risk_level == 'HIGH_FALL_RISK':
            # Use high effectiveness strategies
            for priority, name, params in available_strategies:
                if params['effectiveness'] >= 0.6:
                    return name
            return available_strategies[0][1] if available_strategies else 'crouch'
        elif risk_level == 'MODERATE_FALL_RISK':
            # Use moderate strategies
            for priority, name, params in available_strategies:
                if params['effectiveness'] >= 0.5:
                    return name
            return available_strategies[0][1] if available_strategies else 'arm_swing'
        else:
            # Low risk - no special action needed
            return None
    
    def execute_recovery_strategy(self, strategy_name, robot_state):
        """
        Execute the selected fall recovery strategy
        """
        if strategy_name is None:
            return {'commands': {}, 'status': 'NO_ACTION_NEEDED'}
        
        # Execute based on strategy type
        if strategy_name == 'arm_swing':
            commands = self.execute_arm_swing_strategy(robot_state)
        elif strategy_name == 'hip_strategy':
            commands = self.execute_hip_strategy(robot_state)
        elif strategy_name == 'stepping':
            commands = self.execute_stepping_strategy(robot_state)
        elif strategy_name == 'parachute':
            commands = self.execute_parachute_strategy(robot_state)  # Simulated
        elif strategy_name == 'crouch':
            commands = self.execute_crouch_strategy(robot_state)
        else:
            commands = {}
        
        return {
            'commands': commands,
            'strategy_executed': strategy_name,
            'status': 'EXECUTING_RECOVERY'
        }
    
    def execute_arm_swing_strategy(self, robot_state):
        """
        Execute arm swing strategy to create corrective angular momentum
        """
        # Determine fall direction from IMU data
        imu_data = robot_state.get('imu', {})
        angular_velocity = imu_data.get('angular_velocity', [0, 0, 0])
        
        # Generate arm swing in opposite direction to angular velocity
        commands = {}
        
        # Calculate arm swing torques
        roll_compensation = -angular_velocity[0] * self.arm_swing_torque  # Counter roll
        pitch_compensation = -angular_velocity[1] * self.arm_swing_torque  # Counter pitch
        
        # Apply to both arms with appropriate coordination
        commands.update({
            'left_shoulder_pitch': pitch_compensation,
            'left_shoulder_roll': roll_compensation,
            'left_shoulder_yaw': -angular_velocity[2] * self.arm_swing_torque * 0.5,  # Minor yaw correction
            
            'right_shoulder_pitch': pitch_compensation,
            'right_shoulder_roll': -roll_compensation,  # Opposite for right arm
            'right_shoulder_yaw': angular_velocity[2] * self.arm_swing_torque * 0.5,  # Opposite for right arm
        })
        
        return commands
    
    def execute_hip_strategy(self, robot_state):
        """
        Execute hip-based recovery strategy
        """
        # Shift CoM using hip actuators
        imu_data = robot_state.get('imu', {})
        attitude = imu_data.get('attitude', [1, 0, 0, 0])
        roll, pitch, yaw = self.quaternion_to_euler(attitude)
        
        commands = {}
        
        # Counter tilt with hip torques
        hip_roll_torque = -roll * self.hip_torque_limit * 0.8  # Proportional to tilt
        hip_pitch_torque = -pitch * self.hip_torque_limit * 0.6  # Less aggressive for pitch
        
        commands.update({
            'left_hip_roll': hip_roll_torque,
            'left_hip_pitch': hip_pitch_torque,
            'left_hip_yaw': 0,  # Usually keep yaw neutral
            
            'right_hip_roll': -hip_roll_torque,  # Opposite for balance
            'right_hip_pitch': hip_pitch_torque,  # Same direction
            'right_hip_yaw': 0,
        })
        
        return commands
    
    def execute_stepping_strategy(self, robot_state):
        """
        Execute emergency stepping strategy
        """
        # Determine where to step based on fall direction
        imu_data = robot_state.get('imu', {})
        angular_velocity = imu_data.get('angular_velocity', [0, 0, 0])
        com_position = robot_state.get('com_position', [0, 0, 0.8])
        foot_positions = robot_state.get('foot_positions', {})
        
        # Calculate fall direction and step appropriately
        fall_direction = np.array([angular_velocity[1], -angular_velocity[0]])  # Perpendicular to angular velocity
        fall_direction = fall_direction / (np.linalg.norm(fall_direction) + 1e-6)  # Normalize
        
        # Determine swing foot (the one not currently bearing weight, if known)
        # For simplicity, alternate or use the one that's safer to swing
        swing_foot = 'right'  # Example choice
        
        # Calculate target step position in the direction of fall
        step_distance = 0.4  # meters
        if swing_foot in foot_positions:
            current_pos = foot_positions[swing_foot]
            target_pos = [
                current_pos[0] + fall_direction[0] * step_distance,
                current_pos[1] + fall_direction[1] * step_distance,
                current_pos[2]  # Same height initially
            ]
        else:
            # Default position if foot position unknown
            target_pos = [com_position[0] + fall_direction[0] * step_distance,
                         com_position[1] + fall_direction[1] * step_distance,
                         0.0]  # Ground level
        
        commands = {
            'initiate_emergency_step': True,
            'swing_foot': swing_foot,
            'step_target': target_pos,
            'step_timing': 'immediate',
            'stance_leg_adjustment': 'increase_stiffness',
            'arm_coordination': 'protective_posture'
        }
        
        return commands
    
    def execute_parachute_strategy(self, robot_state):
        """
        Simulated parachute deployment strategy (for safety systems)
        """
        # This is a simulated strategy - in a real robot, this might deploy airbags or similar
        commands = {
            'parachute_deploy': True,
            'airbag_deploy': True,
            'joint_stiffness_reduction': 0.1,  # Reduce to absorb impact
            'protective_posture': 'foetal_position_suggestion'
        }
        
        return commands
    
    def execute_crouch_strategy(self, robot_state):
        """
        Execute crouching strategy to lower CoM and reduce fall impact
        """
        # Lower the body to reduce fall height and impact
        commands = {}
        
        # Flex knees and hips to lower body
        knee_torque = 40.0  # Flex knees
        hip_flex_torque = 30.0  # Flex hips slightly
        
        commands.update({
            'left_knee': knee_torque,
            'right_knee': knee_torque,
            'left_hip_pitch': hip_flex_torque,
            'right_hip_pitch': hip_flex_torque,
            'left_ankle_pitch': -10.0,  # Adjust ankle for balance
            'right_ankle_pitch': -10.0,
        })
        
        # Prepare for impact by reducing joint stiffness
        commands['impact_preparation'] = {
            'stiffness_reduction': 0.5,
            'damping_increase': 2.0,
            'protective_arm_position': 'cover_head'
        }
        
        return commands
    
    def prepare_for_impact(self, robot_state):
        """
        Prepare robot for impact if fall is unavoidable
        """
        commands = {}
        
        # Reduce joint stiffness to absorb impact
        for joint_group in self.impact_absorption_joints:
            commands[f'{joint_group}_stiffness'] = self.impact_absorption_stiffness
            commands[f'{joint_group}_damping'] = self.impact_absorption_damping
        
        # Position limbs to protect vital areas
        commands.update({
            'left_shoulder_pitch': -1.0,  # Cover head/chest
            'right_shoulder_pitch': -1.0,
            'left_elbow': 1.5,  # Bend elbows
            'right_elbow': 1.5,
            'head_neutral_position': True  # Keep head safe
        })
        
        return {
            'commands': commands,
            'status': 'PREPARING_FOR_IMPACT',
            'impact_absorption_active': True
        }
    
    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw)
        """
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.pi / 2 if sinp > 0 else -np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

# Example usage
recovery_system = FallRecoverySystem()

# Example state indicating imminent fall
fall_state = {
    'imu': {
        'attitude': [0.7, 0.2, 0.5, 0.4],  # Highly tilted
        'angular_velocity': [1.5, 1.2, 0.3]  # High angular velocity
    },
    'com_position': [0.08, 0.05, 0.4],  # Low and off-center
    'zmp': [0.25, 0.15],  # Far outside support
    'foot_positions': {
        'left': [0.0, 0.15, 0.0],
        'right': [0.05, -0.15, 0.0]
    },
    'com_velocity': [0.9, 0.4, 0.2]
}

# First detect fall risk
fall_detector = FallDetectionSystem()
fall_result = fall_detector.detect_fall_risk(fall_state)

print(f"Fall Risk Detected: {fall_result['risk_level']}")

# Select and execute recovery strategy
strategy = recovery_system.select_recovery_strategy(fall_result)
print(f"Selected Recovery Strategy: {strategy}")

if strategy:
    recovery_commands = recovery_system.execute_recovery_strategy(strategy, fall_state)
    print(f"Recovery Commands: {recovery_commands}")
else:
    print("No recovery strategy needed")
```

## Conclusion

Bipedal locomotion represents one of the most challenging aspects of humanoid robotics, requiring sophisticated control algorithms to achieve stable and efficient walking. The key components include:

1. **Walking Gaits**: Static and dynamic walking patterns that enable forward motion while maintaining balance
2. **Balance Restoration**: Systems to detect and correct balance disturbances before they lead to falls
3. **Leg Trajectory Planning**: Coordinated movement of swing legs to achieve stable foot placement
4. **Fall Prevention & Recovery**: Proactive systems to avoid falls and reactive systems to mitigate consequences when falls are imminent

Successful implementation of these components requires tight integration between perception, planning, and control systems. Modern humanoid robots continue to improve in their walking capabilities, but challenges remain in achieving human-like efficiency and adaptability to diverse terrains and conditions.

The field continues to evolve with advances in machine learning, allowing robots to learn from experience and adapt their gait patterns to different situations and environments.