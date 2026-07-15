"""
Evaluation script for trained PPO agent.

⚠️ IMPORTANT: This script is a Webots external controller and must run:
   1. With Webots 2023b+ installed and running
   2. With a world file that has the robot configured with controller: "<extern>"
   3. In a terminal WHILE Webots simulation is playing

Usage:
   1. Open Webots and load your world file (.wbt)
   2. Set robot controller to: <extern>
   3. In terminal: cd controllers/RL && python evaluate.py --model ../models/ppo_agent
   4. Press Play button in Webots


This script:
1. Loads a trained PPO model
2. Runs inference on the environment
3. Collects and displays statistics
4. Logs episode results

Command-line usage:
    python evaluate.py --model MODEL_PATH [--episodes NUM_EPISODES] [--stochastic]
    ex :  python evaluate.py --model ../models/ppo_agent --episodes 10
"""

# IMPORTANT: Load .env variables BEFORE importing anything that depends on Webots
import os
import sys
from pathlib import Path

# Load .env file from repository root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                os.environ[key] = value

                if key == "PYTHONPATH":
                    for p in value.split(os.pathsep):
                        p = p.strip()
                        if p and p not in sys.path:
                            sys.path.insert(0, p)
                            
import argparse
from typing import Dict, Any, List
import numpy as np
from stable_baselines3 import PPO

from env import WebotsGymEnvironment
from config import EVAL_NUM_EPISODES, EVAL_DETERMINISTIC, EVAL_LOG_FREQUENCY, PPO_DEVICE


class EvaluationStatistics:
    """Tracks and computes evaluation statistics."""

    def __init__(self):
        """Initialize statistics tracker."""
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_distances: List[float] = []
        self.episode_crashes: List[int] = []

    def add_episode(self, reward: float, length: int, distance: float, crashes: int) -> None:
        """
        Add episode statistics.
        
        Args:
            reward: Total episode reward
            length: Episode length in steps
            distance: Total distance traveled
            crashes: Number of crashes in episode
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_distances.append(distance)
        self.episode_crashes.append(crashes)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.episode_rewards:
            return {}

        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "mean_episode_length": np.mean(self.episode_lengths),
            "mean_distance": np.mean(self.episode_distances),
            "total_crashes": np.sum(self.episode_crashes),
            "crash_rate": np.mean(self.episode_crashes),
            "num_episodes": len(self.episode_rewards),
        }

    def print_summary(self) -> None:
        """Print summary statistics to console."""
        summary = self.get_summary()
        
        if not summary:
            print("No episodes completed")
            return

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nNumber of episodes: {summary['num_episodes']}")
        print(f"\nReward Statistics:")
        print(f"  Mean reward:     {summary['mean_reward']:8.2f}")
        print(f"  Std reward:      {summary['std_reward']:8.2f}")
        print(f"  Min reward:      {summary['min_reward']:8.2f}")
        print(f"  Max reward:      {summary['max_reward']:8.2f}")
        print(f"\nPerformance Metrics:")
        print(f"  Mean episode length: {summary['mean_episode_length']:8.1f} steps")
        print(f"  Mean distance:       {summary['mean_distance']:8.2f} m")
        print(f"  Total crashes:       {summary['total_crashes']:8.0f}")
        print(f"  Crash rate:          {summary['crash_rate']:8.2%}")
        print("=" * 70 + "\n")

    def save_results(self, filepath: str) -> None:
        """
        Save results to file.
        
        Args:
            filepath: Path to save results
        """
        summary = self.get_summary()
        
        with open(filepath, "w") as f:
            f.write("Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n\nPer-Episode Results:\n")
            f.write("-" * 50 + "\n")
            f.write("Episode | Reward    | Steps | Distance | Crashes\n")
            f.write("-" * 50 + "\n")
            
            for i, (r, l, d, c) in enumerate(zip(
                self.episode_rewards,
                self.episode_lengths,
                self.episode_distances,
                self.episode_crashes,
            )):
                f.write(f"{i+1:7d} | {r:9.2f} | {l:5d} | {d:8.2f} | {c:7d}\n")


def evaluate_model(model: PPO, env: WebotsGymEnvironment,
                   num_episodes: int = EVAL_NUM_EPISODES,
                   deterministic: bool = EVAL_DETERMINISTIC,
                   log_frequency: int = EVAL_LOG_FREQUENCY) -> EvaluationStatistics:
    """
    Evaluate trained model on environment.
    
    Args:
        model: Trained PPO model
        env: Webots RL environment
        num_episodes: Number of episodes to run
        deterministic: Use deterministic policy (no exploration)
        log_frequency: Print progress every N steps
        
    Returns:
        EvaluationStatistics object with results
    """
    print("\n" + "=" * 70)
    print("STARTING EVALUATION")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Deterministic: {deterministic}")
    print()
    
    # Disable domain randomization for evaluation
    env.disable_domain_randomization()
    
    stats = EvaluationStatistics()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        print(f"Episode {episode + 1}/{num_episodes}: ", end="", flush=True)
        
        step_count = 0
        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            step_count += 1
            done = done or truncated
            
            # Periodic logging
            if step_count % log_frequency == 0:
                distance = info.get("total_distance", 0.0)
                print(f"\n  Step {step_count}: Reward={episode_reward:.2f}, "
                      f"Distance={distance:.2f}m, Speed={info.get('vitesse_ms', 0):.2f}m/s", 
                      end="", flush=True)
        
        # Get final episode statistics
        episode_stats = env.get_episode_statistics()
        total_distance = episode_stats.get("total_distance", 0.0)
        crashes = episode_stats.get("crashes_this_episode", 0)
        
        print(f"\n  Episode reward: {episode_reward:.2f}")
        print(f"  Episode steps:  {episode_steps}")
        print(f"  Distance:       {total_distance:.2f}m")
        print(f"  Crashes:        {crashes}")
        print()
        
        # Record statistics
        stats.add_episode(episode_reward, episode_steps, total_distance, crashes)
    
    return stats


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_NUM_EPISODES,
        help=f"Number of evaluation episodes (default: {EVAL_NUM_EPISODES})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (optional)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    
    args = parser.parse_args()
    
    # Check model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try with .zip extension
        model_path_zip = Path(str(args.model) + ".zip")
        if model_path_zip.exists():
            args.model = str(model_path_zip)
        else:
            print(f"ERROR: Model file not found: {args.model}")
            return
    
    print("=" * 70)
    print("AUTONOMOUS DRIVING RL - EVALUATION")
    print("=" * 70)
    
    # Create environment
    print("\nCreating environment...")
    env = WebotsGymEnvironment(enable_randomization=False)
    print("Environment created!")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model = PPO.load(args.model, env=env, device=PPO_DEVICE)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        env.close()
        return
    
    # Run evaluation
    deterministic = not args.stochastic
    stats = evaluate_model(
        model,
        env,
        num_episodes=args.episodes,
        deterministic=deterministic,
        log_frequency=EVAL_LOG_FREQUENCY,
    )
    
    # Print and save results
    stats.print_summary()
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats.save_results(str(output_path))
        print(f"Results saved to: {args.output}")
    
    env.close()


if __name__ == "__main__":
    main()
