"""
CoVAPSy RL Agent Package.

Autonomous driving reinforcement learning framework for Webots.

Modules:
- config: All constants and hyperparameters
- env: Gymnasium environment
- lidar: LiDAR sensor management
- reward: Reward computation
- domain_randomization: Parameter randomization for robust training
- callbacks: Training callbacks
- utils: Utility functions
- train: Training script
- evaluate: Evaluation script
"""

from .env import WebotsGymEnvironment
from .config import *
from .lidar import LidarManager
from .reward import RewardManager
from .domain_randomization import DomainRandomizer

__version__ = "1.0.0"
__author__ = "CoVAPSy Team"

__all__ = [
    "WebotsGymEnvironment",
    "LidarManager",
    "RewardManager",
    "DomainRandomizer",
]
