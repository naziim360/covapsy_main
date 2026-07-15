"""
Callbacks for RL training.
"""

from typing import Dict
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback 


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback to log training progress and statistics.

    Called after each environment step during training.
    """

    def __init__(self, log_frequency: int = 1000, verbose: int = 1):
        """
        Initialize callback.

        Args:
            log_frequency: Log every N steps
            verbose: Verbosity level (0-2)
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.last_logged_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """
        Called after each step.

        Returns:
            True to continue training, False to stop
        """
        self.current_episode_length += 1

        # SB3 populates self.locals with vectorized ("dones"/"rewards", plural)
        # keys from collect_rollouts — assumes a single env (n_envs=1).
        dones = self.locals.get("dones", [False])
        rewards = self.locals.get("rewards", [0.0])

        self.current_episode_reward += rewards[0]

        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if self.num_timesteps - self.last_logged_step >= self.log_frequency:
                self._log_statistics()
                self.last_logged_step = self.num_timesteps

            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True

    def _log_statistics(self) -> None:
        """Log training statistics to console."""
        if not self.episode_rewards:
            return

        avg_reward = np.mean(self.episode_rewards[-10:])
        avg_length = np.mean(self.episode_lengths[-10:])
        max_reward = np.max(self.episode_rewards[-10:])

        if self.verbose >= 1:
            print(
                f"\nStep {self.num_timesteps}: "
                f"Avg Reward (last 10): {avg_reward:.2f}, "
                f"Avg Episode Length: {avg_length:.0f}, "
                f"Max Reward: {max_reward:.2f}"
            )

    def get_statistics(self) -> Dict[str, float]:
        """
        Get current training statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
        }