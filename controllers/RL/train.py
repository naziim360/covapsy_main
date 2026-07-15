"""
Training script for PPO agent on autonomous driving task.

⚠️ IMPORTANT: This script is a Webots external controller and must run:
   1. With Webots 2023b+ installed and running
   2. With a world file that has the robot configured with controller: "<extern>"
   3. In a terminal WHILE Webots simulation is playing

Usage:
   1. Open Webots and load your world file (.wbt)
   2. Set robot controller to: <extern>
   3. In terminal: cd controllers/RL && python train.py --steps 200000
   4. Press Play button in Webots

For setup details, see: WEBOTS_SETUP.md

This script:
1. Creates the Webots RL environment
2. Initializes or loads PPO model
3. Trains for specified timesteps
4. Saves trained model
5. Logs progress to TensorBoard

Command-line usage:
    python train.py [--model MODEL_PATH] [--steps TOTAL_STEPS]
"""

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

import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env import WebotsGymEnvironment
from config import (
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_MAX_GRAD_NORM,
    PPO_VERBOSE,
    PPO_DEVICE,
    TENSORBOARD_LOG_DIR,
    TRAINING_TOTAL_TIMESTEPS,
    MODELS_DIR,
    DEFAULT_MODEL_NAME,
)
from callbacks import TrainingProgressCallback, CheckpointCallback


def create_environment() -> WebotsGymEnvironment:
    """
    Create and verify the Webots RL environment.
    
    Returns:
        WebotsGymEnvironment instance
        
    Raises:
        RuntimeError: If environment verification fails
    """
    print("=" * 70)
    print("AUTONOMOUS DRIVING RL - TRAINING")
    print("=" * 70)
    
    print("\n Creating environment...")
    env = WebotsGymEnvironment(enable_randomization=True)
    
    print(" Verifying Gymnasium compatibility...")
    check_env(env)
    print(" Environment verification passed!")
    
    return env


def create_or_load_model(env: WebotsGymEnvironment, 
                        model_path: str = None,
                        new_learning_rate: float = None) -> PPO:
    """
    Create new PPO model or load existing one.
    
    Args:
        env: Webots RL environment
        model_path: Path to existing model to load (optional)
        new_learning_rate: Override learning rate (optional)
        
    Returns:
        PPO model instance
    """
    lr = new_learning_rate or PPO_LEARNING_RATE
    
    if model_path:
        print(f"\n Loading model from: {model_path}")
        try:
            model = PPO.load(model_path, env=env, device=PPO_DEVICE)
            if new_learning_rate:
                model.learning_rate = new_learning_rate
            print(" Model loaded successfully!")
            return model
        except FileNotFoundError:
            print(f" ERROR: Model file not found: {model_path}")
            print(" Creating new model instead...")
    
    print("\n Creating new PPO model...")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=lr,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        verbose=PPO_VERBOSE,
        device=PPO_DEVICE,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )
    print(" New model created!")
    
    return model


def train_model(model: PPO, env: WebotsGymEnvironment,
               total_timesteps: int = TRAINING_TOTAL_TIMESTEPS,
               save_path: str = None,
               checkpoint_freq: int = None) -> PPO:
    """
    Train PPO model for specified timesteps.

    Args:
        model: PPO model to train
        env: Webots RL environment
        total_timesteps: Total timesteps to train for
        save_path: Path to save trained model
        checkpoint_freq: Save an intermediate checkpoint every N timesteps

    Returns:
        Trained PPO model
    """
    print(f"\n Starting training for {total_timesteps:,} timesteps...")
    print(f" Learning rate: {model.learning_rate}")
    print(f" Batch size: {PPO_BATCH_SIZE}")
    print(f" N steps: {PPO_N_STEPS}")
    print()

    start_time = time.time()

    # Progress logging callback
    progress_callback = TrainingProgressCallback(log_frequency=5000, verbose=1)
    callbacks = [progress_callback]

    # Checkpoint callback (optional)
    if checkpoint_freq:
        checkpoint_dir = Path(MODELS_DIR) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix=DEFAULT_MODEL_NAME,
        )
        callbacks.append(checkpoint_callback)
        print(f" Checkpointing every {checkpoint_freq:,} steps to: {checkpoint_dir}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        print("\n Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
    except Exception as e:
        print(f"\n ERROR during training: {e}")
        raise

    elapsed_time = time.time() - start_time
    print(f" Training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f" Model saved to: {save_path}")

    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train PPO agent for autonomous driving")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to existing model to load and continue training",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=TRAINING_TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TRAINING_TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--checkpoints_freq",
        type=int,
        default=None,
        help=f"Checkpoint frequency in timesteps (default: no intermediate checkpoints)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = create_environment()
    
    # Create or load model
    model = create_or_load_model(
        env,
        model_path=args.model,
        new_learning_rate=args.lr,
    )
    
    # Determine save path
    save_path = args.output
    if save_path is None:
        models_dir = Path(MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = str(models_dir / f"{DEFAULT_MODEL_NAME}_{timestamp}")
    
    
    # Train model
    model = train_model(
        model,
        env,
        total_timesteps=args.steps,
        save_path=save_path,
        checkpoint_freq=args.checkpoints_freq,
    )    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved at: {save_path}")
    print("\nTo evaluate the model:")
    print(f"  python evaluate.py --model {save_path}")
    
    env.close()


if __name__ == "__main__":
    main()
