"""
Checkpoint and milestone tracking module for Webots supervisor.

Manages checkpoint detection, validation, and tracking logic.
"""

import math
from typing import List, Tuple


class Checkpoint:
    """
    Represents a checkpoint/milestone in the race track.
    
    Attributes:
        center_x: X coordinate of checkpoint center
        center_y: Y coordinate of checkpoint center
        radius: Checkpoint detection radius
    """

    def __init__(self, center_x: float, center_y: float, radius: float = 1.0):
        """
        Initialize checkpoint.
        
        Args:
            center_x: X coordinate
            center_y: Y coordinate
            radius: Detection radius (default 1.0 meter)
        """
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def is_point_inside(self, x: float, y: float) -> bool:
        """
        Check if point is inside checkpoint.
        
        Args:
            x: Point X coordinate
            y: Point Y coordinate
            
        Returns:
            True if point is within checkpoint radius
        """
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance <= self.radius

    def distance_to_point(self, x: float, y: float) -> float:
        """
        Calculate distance from checkpoint center to point.
        
        Args:
            x: Point X coordinate
            y: Point Y coordinate
            
        Returns:
            Euclidean distance in meters
        """
        dx = x - self.center_x
        dy = y - self.center_y
        return math.sqrt(dx*dx + dy*dy)


class CheckpointTracker:
    """
    Tracks vehicle progress through checkpoints.
    
    Maintains:
    - Current checkpoint index
    - Lap counter
    - Visited checkpoints in current lap
    - Best lap time
    """

    def __init__(self, checkpoints: List[Checkpoint]):
        """
        Initialize checkpoint tracker.
        
        Args:
            checkpoints: List of Checkpoint objects
        """
        self.checkpoints = checkpoints
        self.current_checkpoint_index = 0
        self.lap_count = 0
        self.visited_this_lap = set()
        self.best_lap_time = None
        self.current_lap_start_time = 0.0

    def check_progress(self, x: float, y: float, 
                      simulation_time: float) -> Tuple[bool, int]:
        """
        Check vehicle progress and update checkpoint tracking.
        
        Returns whether vehicle crossed into a checkpoint and the checkpoint index.
        
        Args:
            x: Vehicle X coordinate
            y: Vehicle Y coordinate
            simulation_time: Current simulation time (seconds)
            
        Returns:
            (checkpoint_crossed, checkpoint_index) tuple
        """
        # Check current checkpoint
        current_cp = self.checkpoints[self.current_checkpoint_index]
        
        if current_cp.is_point_inside(x, y):
            # Already visited this checkpoint in current lap
            if self.current_checkpoint_index in self.visited_this_lap:
                return False, self.current_checkpoint_index
            
            # New checkpoint visited
            self.visited_this_lap.add(self.current_checkpoint_index)
            
            # Check if lap complete (returned to first checkpoint)
            if self.current_checkpoint_index == 0 and len(self.visited_this_lap) > 1:
                lap_time = simulation_time - self.current_lap_start_time
                self.lap_count += 1
                
                # Update best lap
                if self.best_lap_time is None or lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                
                # Reset for new lap
                self.visited_this_lap.clear()
                self.current_lap_start_time = simulation_time
            
            # Advance to next checkpoint
            self.current_checkpoint_index = (self.current_checkpoint_index + 1) % len(self.checkpoints)
            
            return True, self.current_checkpoint_index
        
        return False, self.current_checkpoint_index

    def reset(self, simulation_time: float = 0.0) -> None:
        """
        Reset checkpoint tracking for new episode.
        
        Args:
            simulation_time: Current simulation time
        """
        self.current_checkpoint_index = 0
        self.lap_count = 0
        self.visited_this_lap.clear()
        self.current_lap_start_time = simulation_time

    def get_progress(self) -> float:
        """
        Get current lap progress as fraction (0.0 to 1.0).
        
        Returns:
            Fraction of lap completed
        """
        if not self.checkpoints:
            return 0.0
        return len(self.visited_this_lap) / len(self.checkpoints)

    def get_lap_count(self) -> int:
        """Get number of completed laps."""
        return self.lap_count

    def get_best_lap_time(self) -> float:
        """
        Get best lap time recorded.
        
        Returns:
            Best lap time in seconds, or None if no lap completed
        """
        return self.best_lap_time

    def get_current_checkpoint_index(self) -> int:
        """Get index of current target checkpoint."""
        return self.current_checkpoint_index


# Default checkpoints for training track
DEFAULT_CHECKPOINTS_TRAIN = [
    Checkpoint(0.0, 0.0, 1.5),
    Checkpoint(3.0, 3.0, 1.5),
    Checkpoint(-3.0, 3.0, 1.5),
]

# Default checkpoints for test track
DEFAULT_CHECKPOINTS_TEST = [
    Checkpoint(2.5, 5.5, 1.5),
    Checkpoint(5.5, 0.0, 1.5),
    Checkpoint(0.0, -4.5, 1.5),
    Checkpoint(-3.0, -0.5, 1.5),
]
