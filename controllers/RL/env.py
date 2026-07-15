"""
Gymnasium environment for autonomous driving in Webots.

⚠️ WEBOTS CONTROLLER: This module requires Webots to be running.
   The 'vehicle' and 'controller' modules are Webots APIs.
   This environment cannot run outside of Webots.

This module implements the core RL environment that interfaces with:
- Webots simulation through Vehicle Driver API
- LiDAR sensor via LidarManager
- Reward computation via RewardManager
- Domain randomization via DomainRandomizer
- Vehicle control via SpeedController

The environment follows the Gymnasium API standard.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from vehicle import Driver  # type: ignore
from controller import Lidar  # type: ignore

import config
from lidar import LidarManager
from reward import RewardManager
from domain_randomization import DomainRandomizer
from utils import (
    SpeedController,
    VehicleController,
    ObservationBuilder,
    create_info_dict,
)


class WebotsGymEnvironment(Driver, gym.Env):
    """
    Gymnasium environment for autonomous driving RL agent.
    
    This environment:
    - Interfaces with Webots simulation
    - Manages LiDAR sensor acquisition
    - Computes multi-component rewards
    - Applies domain randomization
    - Handles episode resets and vehicle repositioning
    - Provides observations in Gymnasium Dict format
    
    Action: Single steering angle [-1, 1] (normalized)
    Speed: Controlled by P-corrector based on obstacle distance
    Observation: Dict with LiDAR scans, speed, and steering history
    Reward: Multi-component reward (distance, speed, centering, smoothness)
    """

    def __init__(self, enable_randomization: bool = config.ENABLE_DOMAIN_RANDOMIZATION):
        """
        Initialize the autonomous driving environment.
        
        Args:
            enable_randomization: Whether to enable domain randomization
        """
        # Initialize Webots Driver
        super().__init__()

        # ====================================================================
        # VEHICLE STATE
        # ====================================================================
        self.steering_angle_deg: float = 0.0
        self.previous_steering_angle_deg: float = 0.0
        self.commanded_speed_ms: float = config.VITESSE_MIN_M_S

        # ====================================================================
        # EPISODE METRICS
        # ====================================================================
        self.reset_counter: int = 0
        self.current_episode_crashes: int = 0
        self.total_crashes: int = 0
        self.packet_number: int = 0

        # ====================================================================
        # WEBOTS DEVICES
        # ====================================================================
        self.emitter = super().getDevice("emitter")
        self.receiver = super().getDevice("receiver")
        self.receiver.enable(config.RECEIVER_SAMPLING_PERIOD)

        lidar_device = super().getDevice("RpLidarA2")
        timestep_ms = int(super().getBasicTimeStep())
        lidar_device.enable(timestep_ms)
        lidar_device.enablePointCloud()

        self.timestep_s: float = timestep_ms / 1000.0

        # ====================================================================
        # MANAGERS
        # ====================================================================
        self.lidar_manager = LidarManager(
            lidar_device=lidar_device,
            enable_noise=config.ENABLE_LIDAR_NOISE,
            enable_dropout=config.ENABLE_LIDAR_DROPOUT,
            noise_std=config.LIDAR_NOISE_STD,
            dropout_prob=config.LIDAR_DROPOUT_PROB,
        )

        self.reward_manager = RewardManager()

        self.domain_randomizer = DomainRandomizer(enable=enable_randomization)

        self.speed_controller = SpeedController(
            kp=config.KP_VITESSE,
            target_distance_m=config.DISTANCE_CIBLE_M,
            min_safety_speed=config.VITESSE_SECURITE_MIN,
        )

        # ====================================================================
        # CURRENT OBSERVATION
        # ====================================================================
        self.current_observation: Optional[Dict[str, np.ndarray]] = None

        # ====================================================================
        # GYMNASIUM SPACES
        # ====================================================================
        
        # Action space: steering angle normalized to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: Dictionary with LiDAR, speed, and angle history
        self.observation_space = gym.spaces.Dict({
            "current_lidar": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.LIDAR_SECTOR_SIZE,),
                dtype=np.float32,
            ),
            "previous_lidar": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.LIDAR_SECTOR_SIZE,),
                dtype=np.float32,
            ),
            "current_speed": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
            "previous_angle": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
        })

    # ========================================================================
    # ENVIRONMENT STEP
    # ========================================================================

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Process:
        1. Parse steering action
        2. Apply domain randomization to steering
        3. Compute speed based on P-corrector
        4. Apply commands to vehicle
        5. Step simulation
        6. Get new observation and compute reward
        
        Args:
            action: Steering command, normalized to [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # ====================================================================
        # 1. PARSE AND PROCESS ACTION
        # ====================================================================
        steering_normalized = float(action[0])
        self.steering_angle_deg = steering_normalized * config.MAXANGLE_DEGRE
        self.steering_angle_deg = np.clip(
            self.steering_angle_deg, -config.MAXANGLE_DEGRE, config.MAXANGLE_DEGRE
        )

        # Apply domain randomization to steering
        steering_randomized = self.domain_randomizer.apply_steering_randomization(
            self.steering_angle_deg
        )
        steering_randomized = np.clip(
            steering_randomized, -config.MAXANGLE_DEGRE, config.MAXANGLE_DEGRE
        )

        # ====================================================================
        # 2. COMPUTE SPEED USING P-CORRECTOR
        # ====================================================================
        if self.current_observation is not None:
            front_sector = self.current_observation["current_lidar"][80:120]
            min_distance = np.min(front_sector)
        else:
            min_distance = 1.0

        # Apply domain randomization to controller gain
        kp_randomized = self.domain_randomizer.apply_controller_randomization(
            config.KP_VITESSE
        )
        self.speed_controller.set_gain(kp_randomized)

        self.commanded_speed_ms = self.speed_controller.compute_speed(min_distance)

        # Apply domain randomization to speed
        speed_randomized = self.domain_randomizer.apply_speed_randomization(
            self.commanded_speed_ms
        )

        # ====================================================================
        # 3. APPLY COMMANDS
        # ====================================================================
        VehicleController.apply_commands(self, speed_randomized, steering_randomized)

        # ====================================================================
        # 4. STEP SIMULATION
        # ====================================================================
        super().step()

        # ====================================================================
        # 5. GET OBSERVATION AND COMPUTE REWARD
        # ====================================================================
        observation = self._get_observation()
        reward, terminated = self._compute_reward_and_termination()

        # Update angle for next step's smoothness penalty
        self.previous_steering_angle_deg = self.steering_angle_deg
        self.reset_counter += 1

        # ====================================================================
        # 6. PREPARE INFO
        # ====================================================================
        front_sector = observation["current_lidar"][60:141]
        min_distance_step = np.min(front_sector)

        info = create_info_dict(
            speed_ms=self.commanded_speed_ms,
            angle_deg=self.steering_angle_deg,
            min_distance_norm=min_distance_step,
            total_distance=self.reward_manager.get_total_distance(),
            episode_step=self.reset_counter,
            crash_count=self.total_crashes,
        )

        return observation, reward, terminated, False, info

    # ========================================================================
    # OBSERVATION
    # ========================================================================

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Acquire and construct current observation.
        
        Returns:
            Observation dictionary
        """
        lidar_current = self.lidar_manager.get_lidar_with_retry()

        if self.current_observation is None:
            # Initial observation
            obs = ObservationBuilder.build_observation(
                current_lidar=lidar_current,
                previous_lidar=lidar_current.copy(),
                current_speed_ms=0.0,
                previous_angle_deg=0.0,
                is_initial=True,
            )
        else:
            obs = ObservationBuilder.build_observation(
                current_lidar=lidar_current,
                previous_lidar=self.current_observation["current_lidar"],
                current_speed_ms=self.commanded_speed_ms,
                previous_angle_deg=self.previous_steering_angle_deg,
                is_initial=False,
            )

        self.current_observation = obs
        return obs

    # ========================================================================
    # REWARD AND TERMINATION
    # ========================================================================

    def _compute_reward_and_termination(self) -> Tuple[float, bool]:
        """
        Compute reward and determine episode termination.
        
        Returns:
            (reward, terminated) tuple
        """
        if self.current_observation is None:
            return 0.0, False


        raw_for_collision_detection = self.lidar_manager.get_lidar_cleaned()
        
        # Compute reward
        reward, terminated = self.reward_manager.compute_reward(
            observation=self.current_observation,
            current_angle=self.steering_angle_deg,
            timestep_s=self.timestep_s,
            lidar_problem_count=self.lidar_manager.get_problem_count(),
            lidar_raw=raw_for_collision_detection
        )

        # # Check termination conditions
        # terminated = False

        # Condition 1: Episode length exceeded
        episode_step_limit = self.domain_randomizer.get_episode_step_limit(
            config.RESET_STEP
        )
        if self.reset_counter >= episode_step_limit:
            terminated = True
            print(
                f" Episode terminated - Steps: {self.reset_counter}, "
                f"Distance: {self.reward_manager.get_total_distance():.1f}m"
            )

        # Condition 2: Too many LiDAR acquisition problems
        if self.lidar_manager.get_problem_count() > config.LIDAR_PROBLEM_THRESHOLD:
            terminated = True

        # Track crashes
        if reward == config.REWARD_COLLISION_PENALTY:
            self.current_episode_crashes += 1
            self.total_crashes += 1

        return reward, terminated

    # ========================================================================
    # RESET
    # ========================================================================

    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Process:
        1. Reset vehicle commands and state
        2. Stabilize simulation
        3. Request repositioning from supervisor (if crashed)
        4. Apply new domain randomization parameters
        5. Get initial observation
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            (initial_observation, info) tuple
        """
        # ====================================================================
        # 1. RESET VEHICLE STATE
        # ====================================================================
        self.steering_angle_deg = 0.0
        self.previous_steering_angle_deg = 0.0
        self.commanded_speed_ms = 0.0

        VehicleController.apply_commands(self, 0.0, 0.0)

        # ====================================================================
        # 2. STABILIZE SIMULATION
        # ====================================================================
        for _ in range(20):
            super().step()

        # ====================================================================
        # 3. HANDLE CRASH REPOSITIONING
        # ====================================================================
        if self.current_episode_crashes > 0:
            self._handle_crash_reset()

        # ====================================================================
        # 4. RANDOMIZE EPISODE PARAMETERS
        # ====================================================================
        self.domain_randomizer.randomize_episode_parameters()
        
        # Apply randomization to controller
        kp_randomized = self.domain_randomizer.apply_controller_randomization(
            config.KP_VITESSE
        )
        self.speed_controller.set_gain(kp_randomized)

        # ====================================================================
        # 5. RESET MANAGERS
        # ====================================================================
        self.reset_counter = 0
        self.current_episode_crashes = 0
        self.reward_manager.reset()
        self.lidar_manager.reset_problem_counter()

        # ====================================================================
        # 6. GET INITIAL OBSERVATION
        # ====================================================================
        self.current_observation = None  # Force initial observation
        observation = self._get_observation()

        info = {
            "reset_type": "standard" if self.current_episode_crashes == 0 else "crash_recovery",
            "randomization_params": self.domain_randomizer.get_randomization_state(),
        }

        return observation, info

    def _handle_crash_reset(self) -> None:
        """
        Handle repositioning after crash.
        Communicates with supervisor to reset vehicle position.
        """
        # Wait for vehicle to stop (bounded)
        timeout = 0
        while abs(super().getTargetCruisingSpeed()) > 0.001 and timeout < 1000:
            super().step()
            timeout += 1
        if timeout >= 1000:
            print("  Timeout waiting for vehicle to stop before crash reset")

        # Request repositioning
        self.packet_number += 1
        self.emitter.send(f"voiture crash {self.packet_number}".encode("utf-8"))
        super().step()

        # Wait for acknowledgment
        timeout = 0
        while self.receiver.getQueueLength() == 0 and timeout < 1000:
            VehicleController.apply_speed_command(self, 0.0)
            super().step()
            timeout += 1

        if self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            self.receiver.nextPacket()
            print(f" Supervisor response: {data}")
        else:
            print("  Supervisor timeout")

        # Ensure vehicle is stopped (bounded)
        VehicleController.apply_commands(self, 0.0, 0.0)
        timeout = 0
        while abs(super().getTargetCruisingSpeed()) >= 0.001 and timeout < 1000:
            super().step()
            timeout += 1
        if timeout >= 1000:
            print("  Timeout waiting for vehicle to fully stop after crash reset")
        
        
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for current episode.
        
        Returns:
            Dictionary with episode metrics
        """
        return {
            "total_distance": self.reward_manager.get_total_distance(),
            "crashes_this_episode": self.current_episode_crashes,
            "total_crashes": self.total_crashes,
            "steps": self.reset_counter,
            "lidar_problems": self.lidar_manager.get_problem_count(),
        }

    def enable_domain_randomization(self) -> None:
        """Enable domain randomization."""
        self.domain_randomizer.enable_randomization()

    def disable_domain_randomization(self) -> None:
        """Disable domain randomization (e.g., for evaluation)."""
        self.domain_randomizer.disable_randomization()
