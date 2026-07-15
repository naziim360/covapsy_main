"""
Webots Supervisor Package for CoVAPSy.

Handles simulation management and vehicle repositioning for RL training.

Modules:
- supervisor: Main supervisor script
- spawn: Vehicle spawning and positioning
- checkpoints: Checkpoint tracking for progress monitoring
- utils: Utility functions
"""

from .spawn import SpawnManager
from .checkpoints import CheckpointTracker, Checkpoint

__version__ = "1.0.0"
__author__ = "CoVAPSy Team"

__all__ = [
    "SpawnManager",
    "CheckpointTracker",
    "Checkpoint",
]
