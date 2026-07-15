"""
Utility functions for Webots supervisor.

Provides helper functions for angle handling, position checking, etc.
"""

import math
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    angle = angle % (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


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
    return low if value < low else high if value > high else value


def is_position_valid(x: float, y: float, z: float,
                     max_x: float = 20.0, max_y: float = 20.0,
                     max_z: float = 0.1) -> bool:
    """
    Check if position is within valid bounds.
    
    Default bounds represent reasonable track limits.
    
    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate (height)
        max_x: Maximum X magnitude
        max_y: Maximum Y magnitude
        max_z: Maximum Z magnitude (height bounds)
        
    Returns:
        True if position is valid
    """
    return (
        abs(x) <= max_x and
        abs(y) <= max_y and
        abs(z) <= max_z
    )


def get_safe_position() -> Tuple[float, float, float, float]:
    """
    Get a safe reset position (fallback when position becomes invalid).
    
    Returns:
        (x, y, z, angle) tuple for safe position
    """
    return (2.98, 2.0, 0.04, -math.pi/2)


def distance_between_points(x1: float, y1: float, 
                           x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two 2D points.
    
    Args:
        x1: First point X
        y1: First point Y
        x2: Second point X
        y2: Second point Y
        
    Returns:
        Distance in same units as input
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx*dx + dy*dy)


def create_rotation_array(angle_rad: float) -> list:
    """
    Create Webots rotation array for axis-angle representation.
    
    Creates a rotation around Z axis with specified angle.
    
    Args:
        angle_rad: Rotation angle in radians
        
    Returns:
        [axis_x, axis_y, axis_z, angle] array for Webots
    """
    return [0, 0, 1, normalize_angle(angle_rad)]
