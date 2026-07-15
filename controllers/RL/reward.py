"""
Reward computation module for autonomous driving RL agent.

Implements the multi-component reward function that incentivizes:
- Safe driving (distance to obstacles)
- Lane centering
- Speed optimization
- Smooth steering
- Progress through episodes

The reward combines multiple components weighted by hyperparameters
defined in config.py.
"""

from typing import Tuple, Dict, Any
import numpy as np
import config


class RewardManager:
    """
    Computes reward for each step based on environment observations.
    
    Reward components:
    - Distance component: Encourages maintaining safe distance
    - Speed component: Rewards forward progress at appropriate speeds
    - Lane centering: Rewards staying in lane center
    - Heading: Rewards directional alignment
    - Steering smoothness: Penalizes abrupt angle changes
    """

    def __init__(self):
        """Initialize reward manager."""
        self.last_angle: float = 0.0
        self.total_distance: float = 0.0
        self.crash_count: int = 0
        self.lidar_problem_count: int = 0

    def _compute_distance_reward(self, front_min_distance: float) -> float:
        """
        Compute distance-based reward component.
        
        Uses logarithmic scaling to give more weight to small distances.
        Encourages maintaining minimum safe distance.
        
        Args:
            front_min_distance: Minimum normalized distance in front sector [0, 1]
            
        Returns:
            Distance reward component
        """
        # Log scale for better gradient signal at small distances
        mini_log = np.log(front_min_distance * 120 + 1e-6)
        return config.REWARD_WEIGHT_DISTANCE * mini_log

    def _compute_speed_reward(self, current_speed: float, 
                             front_min_distance: float) -> float:
        """
        Compute speed-based reward component.
        
        Encourages forward progress while respecting safe distance.
        Speed reward is modulated by how close obstacles are.
        
        Args:
            current_speed: Current speed normalized [0, 1]
            front_min_distance: Minimum normalized distance in front [0, 1]
            
        Returns:
            Speed reward component
        """
        # Safety factor: reduce speed reward when too close to obstacles
        mini_log = np.log(front_min_distance * 120 + 1e-6)
        safe_speed_factor = np.clip(mini_log / 1.94, 0.0, 1.0)  # 1.94 ≈ ln(7)
        
        # Convert normalized speed to m/s
        speed_ms = current_speed * config.VITESSE_MAX_M_S
        
        # Quadratic reward: encourages moderate speeds
        speed_component = (3 * speed_ms - speed_ms ** 2) * safe_speed_factor
        
        return config.REWARD_WEIGHT_SPEED * speed_component

    def _compute_lane_centering_reward(self, lidar_obs: np.ndarray) -> float:
        """
        Compute lane centering reward component.
        
        Encourages symmetric LiDAR readings (left/right symmetry)
        which indicates car is centered in lane.
        
        Args:
            lidar_obs: Current LiDAR observation (normalized)
            
        Returns:
            Lane centering reward
        """
        # Compare left and right sides for symmetry
        # Left: indices 0-39, Right: indices 161-200 (mirrored)
        left_side = lidar_obs[0:40]
        right_side = lidar_obs[161:201][::-1]
        
        # Symmetric distance error
        center_error = np.mean(np.abs(left_side - right_side))
        
        # Exponential reward for symmetry
        lane_reward = np.exp(-(center_error / 0.1) ** 2)
        
        return config.REWARD_WEIGHT_LANE * lane_reward

    def _compute_heading_reward(self, lidar_obs: np.ndarray) -> float:
        """
        Compute heading/orientation reward component.
        
        Encourages alignment with lane direction through
        symmetry in the central LiDAR sector.
        
        Args:
            lidar_obs: Current LiDAR observation (normalized)
            
        Returns:
            Heading reward
        """
        # Central sector symmetry (around index 100)
        # Using indices 80-121 for both sides for comparison
        left_center = lidar_obs[80:121]
        right_center = lidar_obs[80:121][::-1]
        
        # Heading error
        heading_error = np.mean(np.abs(left_center - right_center))
        
        # Exponential reward
        heading_reward = np.exp(-(heading_error / 0.1) ** 2)
        
        return config.REWARD_WEIGHT_HEADING * heading_reward

    def _compute_steering_smoothness_penalty(self, current_angle: float) -> float:
        """
        Compute penalty for abrupt steering angle changes.
        
        Encourages smooth, gradual steering inputs rather than
        sudden changes.
        
        Args:
            current_angle: Current steering angle in degrees
            
        Returns:
            Steering smoothness penalty (typically negative)
        """
        angle_diff = abs(current_angle - self.last_angle)
        angle_penalty = angle_diff / (2 * config.MAXANGLE_DEGRE)
        
        return -config.REWARD_WEIGHT_ANGLE_PENALTY * angle_penalty

    def _compute_distance_traveled_bonus(self, speed_ms: float, 
                                        timestep_s: float) -> float:
        """
        Compute bonus reward for distance traveled.
        
        Small bonus to encourage forward progress.
        
        Args:
            speed_ms: Current speed in m/s
            timestep_s: Simulation timestep in seconds
            
        Returns:
            Distance traveled bonus
        """
        distance_step = speed_ms * timestep_s
        self.total_distance += distance_step
        return config.REWARD_WEIGHT_DISTANCE_TRAVELED * distance_step

    def _check_collision(self, front_min_distance_mm: float) -> Tuple[bool, float]:
        """
        Detect collision and return penalty if occurred.
        
        Args:
            front_min_distance_mm: Minimum distance in front (in mm)
            
        Returns:
            (is_collision, penalty_reward) tuple
        """
        if front_min_distance_mm < config.SEUIL_COLLISION_MM:
            self.crash_count += 1
            return True, config.REWARD_COLLISION_PENALTY
        return False, 0.0

    def compute_reward(self, observation: Dict[str, np.ndarray],
                      current_angle: float, timestep_s: float,
                      lidar_raw: np.ndarray,
                      lidar_problem_count: int = 0) -> Tuple[float, bool]:
                
        """
        Compute total reward for current step and determine episode termination.
        
        Processes:
        1. Extract front LiDAR sector for obstacle detection
        2. Check for collision
        3. Compute multi-component reward if no collision
        4. Check episode termination conditions
        
        Args:
            observation: Dictionary with keys:
                - 'current_lidar': Current LiDAR scan (normalized)
                - 'current_speed': Current speed (normalized)
            current_angle: Current steering angle in degrees
            timestep_s: Simulation timestep in seconds
            lidar_problem_count: Count of LiDAR acquisition problems
            
        Returns:
            (reward, done) tuple
        """
        reward = 0.0
        done = False
        
        # Extract front sector [-40°, +40°] from LiDAR
        # Indices 60-140 correspond to ±40°
        front_sector_mm = lidar_raw[60:141] * 1000 
        front_min_distance = np.min(front_sector_mm)
        
        # Check for collision
        is_collision, collision_penalty = self._check_collision(front_min_distance)
        
        if is_collision:
            reward = collision_penalty
            done = True  
        else:
            # Compute multi-component reward
            current_speed = observation["current_speed"][0]
            
            distance_reward = self._compute_distance_reward(front_min_distance)
            speed_reward = self._compute_speed_reward(current_speed, front_min_distance)
            lane_reward = self._compute_lane_centering_reward(observation["current_lidar"])
            heading_reward = self._compute_heading_reward(observation["current_lidar"])
            steering_penalty = self._compute_steering_smoothness_penalty(current_angle)
            distance_bonus = self._compute_distance_traveled_bonus(
                current_speed * config.VITESSE_MAX_M_S, 
                timestep_s
            )
            
            # Total reward
            reward = (
                distance_reward +
                speed_reward +
                lane_reward +
                heading_reward +
                steering_penalty +
                distance_bonus
            )
        
        # Update steering angle for next step
        self.last_angle = current_angle
        
        # Update episode counter (internal to track progress)
        self.lidar_problem_count = lidar_problem_count
        
        return reward, done

    def reset(self) -> None:
        """Reset reward manager state for new episode."""
        self.last_angle = 0.0
        self.total_distance = 0.0

    def get_total_distance(self) -> float:
        """Get total distance traveled in current episode."""
        return self.total_distance

    def get_crash_count(self) -> int:
        """Get total crashes during training."""
        return self.crash_count
