"""
Domain randomization module for robust RL training.

Implements randomization of simulation parameters to improve generalization:
- LiDAR noise and dropout
- Steering characteristics (gain, bias)
- Controller parameters (P-gain variations)
- Episode duration variations
- Vehicle speed scaling

This ensures the learned policy is robust to simulation-to-reality gap
and parameter uncertainties.
"""

from typing import Tuple
import numpy as np
import config


class DomainRandomizer:
    """
    Manages domain randomization parameters for each episode.
    
    Randomizes:
    - LiDAR sensor characteristics (noise, dropout)
    - Vehicle steering (gain, bias)
    - Controller gains (P-corrector)
    - Episode length variations
    - Vehicle speed scaling
    
    Each episode receives a new random sample of these parameters.
    """

    def __init__(self, enable: bool = config.ENABLE_DOMAIN_RANDOMIZATION):
        """
        Initialize domain randomizer.
        
        Args:
            enable: Whether to enable domain randomization
        """
        self.enable = enable
        self._reset_episode_parameters()

    def _reset_episode_parameters(self) -> None:
        """Reset all randomization parameters for new episode."""
        self.episode_step_limit_multiplier = 1.0
        self.steering_gain_multiplier = 1.0
        self.steering_bias_deg = 0.0
        self.kp_gain_multiplier = 1.0
        self.speed_multiplier = 1.0
        self.lidar_delay_steps = 0

    def randomize_episode_parameters(self) -> None:
        """
        Sample new randomization parameters for the episode.
        
        Should be called at the start of each new episode.
        """
        if not self.enable:
            self._reset_episode_parameters()
            return

        # Randomize episode length
        min_mult, max_mult = config.DOMAIN_RANDOM_EPISODE_LENGTH_RANGE
        self.episode_step_limit_multiplier = np.random.uniform(min_mult, max_mult)

        # Randomize steering characteristics
        min_gain, max_gain = config.DOMAIN_RANDOM_STEERING_GAIN_RANGE
        self.steering_gain_multiplier = np.random.uniform(min_gain, max_gain)

        min_bias, max_bias = config.DOMAIN_RANDOM_STEERING_BIAS_RANGE
        self.steering_bias_deg = np.random.uniform(min_bias, max_bias)

        # Randomize P-controller gain
        min_kp, max_kp = config.DOMAIN_RANDOM_KP_RANGE
        self.kp_gain_multiplier = np.random.uniform(min_kp, max_kp)

        # Randomize vehicle speed
        min_speed, max_speed = config.DOMAIN_RANDOM_SPEED_MULTIPLIER_RANGE
        self.speed_multiplier = np.random.uniform(min_speed, max_speed)

        # Randomize sensor delay (simulation steps)
        delay_min, delay_max = config.DOMAIN_RANDOM_DELAY_RANGE
        self.lidar_delay_steps = np.random.randint(delay_min, delay_max + 1)

    def apply_steering_randomization(self, steering_command_deg: float) -> float:
        """
        Apply randomized steering gain and bias.
        
        Simulates variations in vehicle steering characteristics
        like wear, calibration differences, etc.
        
        Args:
            steering_command_deg: Commanded steering angle in degrees
            
        Returns:
            Steering angle with randomization applied
        """
        if not self.enable:
            return steering_command_deg

        # Apply gain variation
        randomized_angle = steering_command_deg * self.steering_gain_multiplier

        # Apply bias variation
        randomized_angle += self.steering_bias_deg

        return randomized_angle

    def apply_speed_randomization(self, speed_command_ms: float) -> float:
        """
        Apply randomized speed scaling.
        
        Simulates variations in vehicle speed characteristics
        like motor wear, gear calibration, etc.
        
        Args:
            speed_command_ms: Commanded speed in m/s
            
        Returns:
            Speed with randomization applied
        """
        if not self.enable:
            return speed_command_ms

        randomized_speed = speed_command_ms * self.speed_multiplier
        return randomized_speed

    def apply_controller_randomization(self, kp_base: float) -> float:
        """
        Apply randomized P-controller gain.
        
        Simulates tuning uncertainty in the P-corrector.
        
        Args:
            kp_base: Base proportional gain
            
        Returns:
            Gain with randomization applied
        """
        if not self.enable:
            return kp_base

        randomized_kp = kp_base * self.kp_gain_multiplier
        return randomized_kp

    def get_episode_step_limit(self, base_limit: int) -> int:
        """
        Get randomized episode step limit.
        
        Varies episode length to prevent overfitting to
        specific episode duration.
        
        Args:
            base_limit: Base step limit (config.RESET_STEP)
            
        Returns:
            Randomized step limit
        """
        if not self.enable:
            return base_limit

        randomized_limit = int(base_limit * self.episode_step_limit_multiplier)
        return randomized_limit

    def get_sensor_delay_steps(self) -> int:
        """
        Get randomized sensor delay in simulation steps.
        
        Returns:
            Number of steps to delay LiDAR readings
        """
        if not self.enable:
            return 0

        return self.lidar_delay_steps

    def get_randomization_state(self) -> dict:
        """
        Get current randomization parameters for debugging/logging.
        
        Returns:
            Dictionary with all active randomization parameters
        """
        return {
            "episode_step_limit_multiplier": self.episode_step_limit_multiplier,
            "steering_gain_multiplier": self.steering_gain_multiplier,
            "steering_bias_deg": self.steering_bias_deg,
            "kp_gain_multiplier": self.kp_gain_multiplier,
            "speed_multiplier": self.speed_multiplier,
            "lidar_delay_steps": self.lidar_delay_steps,
        }

    def enable_randomization(self) -> None:
        """Enable domain randomization."""
        self.enable = True

    def disable_randomization(self) -> None:
        """Disable domain randomization (use for evaluation)."""
        self.enable = False
        self._reset_episode_parameters()


def create_randomized_config(base_config: dict, randomizer: DomainRandomizer) -> dict:
    """
    Create a configuration dictionary with randomization applied.
    
    Utility function to apply all randomizations at once.
    
    Args:
        base_config: Base configuration dictionary
        randomizer: DomainRandomizer instance
        
    Returns:
        Configuration with randomization applied
    """
    config_copy = base_config.copy()

    if randomizer.enable:
        config_copy["steering_gain"] = randomizer.steering_gain_multiplier
        config_copy["steering_bias"] = randomizer.steering_bias_deg
        config_copy["kp_gain"] = randomizer.kp_gain_multiplier
        config_copy["speed_multiplier"] = randomizer.speed_multiplier
        config_copy["episode_limit_multiplier"] = randomizer.episode_step_limit_multiplier

    return config_copy
