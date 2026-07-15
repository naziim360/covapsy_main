"""
Utility functions for the RL training framework.

Provides helper functions for:
- Vehicle control (speed, steering angle conversion)
- LiDAR data processing
- Observation construction
- Math utilities
"""

from typing import Dict, Tuple, Union
import numpy as np
import math
from vehicle import Driver  # type: ignore
import config


class SpeedController:
    """
    P-corrector for vehicle speed based on obstacle distance.
    
    Maintains a target safe distance from obstacles by modulating speed.
    """

    def __init__(self, kp: float = config.KP_VITESSE,
                 target_distance_m: float = config.DISTANCE_CIBLE_M,
                 min_safety_speed: float = config.VITESSE_SECURITE_MIN):
        """
        Initialize speed controller.
        
        Args:
            kp: Proportional gain
            target_distance_m: Target safe distance in meters
            min_safety_speed: Minimum speed when close to obstacles
        """
        self.kp = kp
        self.target_distance_m = target_distance_m
        self.min_safety_speed = min_safety_speed

    def compute_speed(self, min_distance_normalized: float) -> float:
        """
        Compute vehicle speed based on minimum obstacle distance.
        
        Uses proportional control:
        v = v_min + kp * (d_min - d_target)
        
        Args:
            min_distance_normalized: Minimum normalized distance [0, 1]
            
        Returns:
            Commanded speed in m/s
        """
        # Convert normalized to meters
        min_distance_m = min_distance_normalized * config.LIDAR_RANGE_MM / 1000.0

        # Proportional error
        error_m = min_distance_m - self.target_distance_m

        # Proportional control
        speed_command = self.min_safety_speed + self.kp * error_m

        # Saturation
        speed_command = np.clip(
            speed_command,
            config.VITESSE_MIN_M_S,
            config.VITESSE_MAX_M_S
        )

        return speed_command

    def set_gain(self, kp: float) -> None:
        """Update proportional gain."""
        self.kp = kp

    def set_target_distance(self, distance_m: float) -> None:
        """Update target safe distance."""
        self.target_distance_m = distance_m


class VehicleController:
    """
    Helper for vehicle command application and conversion.
    """

    @staticmethod
    def apply_steering_command(driver: Driver, angle_deg: float) -> None:
        """
        Apply steering angle command to vehicle.
        
        Converts from degrees to radians and applies to Webots driver.
        
        Args:
            driver: Webots Driver instance
            angle_deg: Steering angle in degrees
        """
        angle_rad = -angle_deg * (math.pi / 180.0)
        driver.setSteeringAngle(angle_rad)

    @staticmethod
    def apply_speed_command(driver: Driver, speed_ms: float) -> None:
        """
        Apply speed command to vehicle.
        
        Converts from m/s to km/h and sets as cruising speed.
        
        Args:
            driver: Webots Driver instance
            speed_ms: Speed in m/s
        """
        speed_kmh = speed_ms * 3.6
        driver.setCruisingSpeed(speed_kmh)

    @staticmethod
    def apply_commands(driver: Driver, speed_ms: float, angle_deg: float) -> None:
        """
        Apply both speed and steering commands.
        
        Args:
            driver: Webots Driver instance
            speed_ms: Speed in m/s
            angle_deg: Steering angle in degrees
        """
        VehicleController.apply_speed_command(driver, speed_ms)
        VehicleController.apply_steering_command(driver, angle_deg)


class ObservationBuilder:
    """
    Constructs observation dictionaries for the RL agent.
    """

    @staticmethod
    def build_observation(current_lidar: np.ndarray,
                         previous_lidar: np.ndarray,
                         current_speed_ms: float,
                         previous_angle_deg: float,
                         is_initial: bool = False) -> Dict[str, np.ndarray]:
        """
        Build observation dictionary for RL agent.
        
        Args:
            current_lidar: Current LiDAR measurement (normalized)
            previous_lidar: Previous LiDAR measurement (normalized)
            current_speed_ms: Current speed in m/s
            previous_angle_deg: Previous steering angle in degrees
            is_initial: Whether this is initial observation after reset
            
        Returns:
            Observation dictionary with keys:
            - 'current_lidar': Current scan (shape: (201,))
            - 'previous_lidar': Previous scan (shape: (201,))
            - 'current_speed': Normalized speed (shape: (1,))
            - 'previous_angle': Normalized angle (shape: (1,))
        """
        if is_initial:
            prev_lidar = current_lidar.copy()
            speed_norm = 0.0
            angle_norm = 0.0
        else:
            prev_lidar = previous_lidar
            speed_norm = current_speed_ms / config.VITESSE_MAX_M_S
            angle_norm = previous_angle_deg / config.MAXANGLE_DEGRE

        observation = {
            "current_lidar": current_lidar.astype(np.float32),
            "previous_lidar": prev_lidar.astype(np.float32),
            "current_speed": np.array([speed_norm], dtype=np.float32),
            "previous_angle": np.array([angle_norm], dtype=np.float32),
        }

        return observation

    @staticmethod
    def get_front_sector(lidar_obs: np.ndarray,
                        angle_range_deg: int = 40) -> np.ndarray:
        """
        Extract front sector from LiDAR observation.
        
        Default: [-40°, +40°] from center.
        
        Args:
            lidar_obs: Full LiDAR observation (201 measurements)
            angle_range_deg: Half-angle range (default 40°)
            
        Returns:
            Front sector measurements
        """
        # Full range is 200° (-100° to +100°) with 201 measurements
        # Each degree ≈ 201/200 = 1.005 measurements
        # Center is at index 100
        measurements_per_deg = 201.0 / 200.0

        start_idx = int(100 - angle_range_deg * measurements_per_deg)
        end_idx = int(100 + angle_range_deg * measurements_per_deg)

        return lidar_obs[max(0, start_idx):min(201, end_idx + 1)]


class AngleUtils:
    """
    Utility functions for angle operations.
    """

    @staticmethod
    def normalize_angle(angle_deg: float) -> float:
        """
        Normalize angle to [-180, 180] degrees.
        
        Args:
            angle_deg: Angle in degrees
            
        Returns:
            Normalized angle in degrees
        """
        angle = angle_deg % 360.0
        if angle > 180.0:
            angle -= 360.0
        return angle

    @staticmethod
    def angle_diff(angle1_deg: float, angle2_deg: float) -> float:
        """
        Compute smallest angle difference between two angles.
        
        Args:
            angle1_deg: First angle in degrees
            angle2_deg: Second angle in degrees
            
        Returns:
            Smallest difference in degrees [-180, 180]
        """
        diff = angle1_deg - angle2_deg
        return AngleUtils.normalize_angle(diff)

    @staticmethod
    def clamp_angle(angle_deg: float, max_angle_deg: float) -> float:
        """
        Clamp angle magnitude to maximum.
        
        Args:
            angle_deg: Angle in degrees
            max_angle_deg: Maximum allowed magnitude
            
        Returns:
            Clamped angle
        """
        return np.clip(angle_deg, -max_angle_deg, max_angle_deg)


def create_info_dict(speed_ms: float, angle_deg: float,
                    min_distance_norm: float, total_distance: float,
                    episode_step: int, crash_count: int = 0) -> Dict[str, Union[float, int]]:
    """
    Create info dictionary for step return.
    
    Args:
        speed_ms: Current speed in m/s
        angle_deg: Current steering angle in degrees
        min_distance_norm: Minimum normalized distance
        total_distance: Total distance traveled in episode
        episode_step: Current episode step number
        crash_count: Total crashes so far
        
    Returns:
        Info dictionary
    """
    min_distance_m = min_distance_norm * config.LIDAR_RANGE_MM / 1000.0

    return {
        "vitesse_ms": speed_ms,
        "angle_degre": angle_deg,
        "distance_min_m": min_distance_m,
        "total_distance": total_distance,
        "episode_step": episode_step,
        "crash_count": crash_count,
    }


def clip_value(value: float, low: float, high: float) -> float:
    """
    Clamp value to range [low, high].
    
    Args:
        value: Value to clamp
        low: Lower bound
        high: Upper bound
        
    Returns:
        Clamped value
    """
    return np.clip(value, low, high)
