"""
Vehicle spawning and positioning module for Webots supervisor.

Manages spawn positions and vehicle repositioning logic.
"""

import random
import math
from typing import List, Tuple
from controller import Node, Field  # type: ignore


# Starting positions for each spawn point
# Format: [[x_min, x_max], [y_min, y_max], base_angle]

STARTING_POSITIONS_TRAIN = [
    [[-1.5, 1.5], [-0.225, -0.78], 0.00],  # centre
    [[-3.74, 1.74], [-1.84, -3.13], 0.00],  # en bas
    [[-4.6, -3.7], [-2.5, 2.5], math.pi/2],  # gauche
    [[3.2, 4.5], [3.18, 3.3], math.pi/2]  # haut droite
]

STARTING_POSITIONS_TEST = []


class SpawnManager:
    """
    Manages vehicle spawning and repositioning.
    
    Handles:
    - Selecting random spawn points
    - Randomizing direction of travel
    - Positioning vehicles with noise
    """

    def __init__(self, use_test_track: bool = False):
        """
        Initialize spawn manager.
        
        Args:
            use_test_track: Whether to use test or training track positions
        """
        self.positions = STARTING_POSITIONS_TEST if use_test_track else STARTING_POSITIONS_TRAIN
        self.use_test_track = use_test_track

    def select_random_positions(self, num_positions: int) -> List[int]:
        """
        Select random distinct spawn positions.
        
        Args:
            num_positions: Number of positions to select
            
        Returns:
            List of position indices
        """
        available = min(num_positions, len(self.positions))
        return random.sample(range(len(self.positions)), available)

    def randomize_direction(self) -> int:
        """
        Randomize direction of travel.
        
        Returns:
            0 for normal direction, 1 for reversed
        """
        return random.choice([0, 1])

    def get_spawn_point(self, position_index: int, 
                       direction: int = 0,
                       angle_noise_rad: float = math.pi/12) -> Tuple[float, float, float, float]:
        """
        Get randomized spawn point for a vehicle.
        
        Args:
            position_index: Index into spawn positions list
            direction: 0=normal, 1=reversed
            angle_noise_rad: Angular noise range (±)
            
        Returns:
            (x, y, z, angle) tuple
        """
        if position_index >= len(self.positions):
            raise ValueError(f"Position index {position_index} out of range")
        
        coords = self.positions[position_index]
        
        # Randomize x, y
        x = random.uniform(coords[0][0], coords[0][1])
        y = random.uniform(coords[1][0], coords[1][1])
        z = 0.04  # Constant height
        
        # Randomize angle with noise
        base_angle = coords[2]
        angle = random.uniform(base_angle - angle_noise_rad, base_angle + angle_noise_rad)
        
        # Reverse direction if needed
        if direction == 1:
            angle += math.pi
        
        # Normalize angle to [-pi, pi]
        angle = self._normalize_angle(angle)
        
        return x, y, z, angle

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-pi, pi].
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle
        """
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle


def position_vehicle(node: Node, translation_field: Field, 
                    rotation_field: Field, spawn_x: float, 
                    spawn_y: float, spawn_z: float, 
                    spawn_angle: float) -> None:
    """
    Position a vehicle in the simulation.
    
    Args:
        node: Webots node object
        translation_field: Translation field from node
        rotation_field: Rotation field from node
        spawn_x: X coordinate
        spawn_y: Y coordinate
        spawn_z: Z coordinate
        spawn_angle: Rotation angle in radians
    """
    translation_field.setSFVec3f([spawn_x, spawn_y, spawn_z])
    rotation_field.setSFRotation([0, 0, 1, spawn_angle])
