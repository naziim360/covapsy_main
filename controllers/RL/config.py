"""
Configuration module for CoVAPSy RL Agent.

This module centralizes all constants and hyperparameters used throughout
the autonomous driving RL training framework.
"""

# ============================================================================
# VEHICLE DYNAMICS
# ============================================================================
"""Maximum vehicle speed in m/s (28 km/h converted)."""
VITESSE_MAX_M_S: float = 28.0 / 3.6  # 28 km/h → 7.78 m/s

"""Minimum vehicle speed to avoid complete stop (m/s)."""
VITESSE_MIN_M_S: float = 0.2

"""Maximum steering angle in degrees."""
MAXANGLE_DEGRE: float = 16.0

# ============================================================================
# LIDAR CONFIGURATION
# ============================================================================
"""LiDAR maximum range in millimeters."""
LIDAR_RANGE_MM: int = 12000

"""Number of LiDAR measurements in the selected sector ([-100°, +100°])."""
LIDAR_SECTOR_SIZE: int = 201

"""LiDAR collision threshold (normalized distance)."""
SEUIL_COLLISION_MM: float = 300

"""Maximum retry attempts for LiDAR data acquisition."""
MAX_LIDAR_RETRY: int = 50

# ============================================================================
# SPEED CONTROLLER (P-CORRECTOR) PARAMETERS
# ============================================================================
"""Proportional gain for speed control based on minimum distance."""
KP_VITESSE: float = 0.8

"""Target safety distance from obstacles (meters)."""
DISTANCE_CIBLE_M: float = 0.8

"""Minimum speed when close to obstacles (m/s)."""
VITESSE_SECURITE_MIN: float = 0.4

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
"""Receiver sampling period in milliseconds."""
RECEIVER_SAMPLING_PERIOD: int = 64

"""Maximum number of simulation steps per episode."""
RESET_STEP: int = 2000

# ============================================================================
# REWARD FUNCTION WEIGHTS
# ============================================================================
"""Weight for distance-based reward component."""
REWARD_WEIGHT_DISTANCE: float = 1.0

"""Weight for speed reward component."""
REWARD_WEIGHT_SPEED: float = 3.0

"""Weight for lane centering reward component."""
REWARD_WEIGHT_LANE: float = 3.0

"""Weight for heading/orientation reward component."""
REWARD_WEIGHT_HEADING: float = 1.5

"""Weight for angle change penalty."""
REWARD_WEIGHT_ANGLE_PENALTY: float = 2.0

"""Weight for distance traveled bonus."""
REWARD_WEIGHT_DISTANCE_TRAVELED: float = 0.1

"""Penalty for collision."""
REWARD_COLLISION_PENALTY: float = -100.0

"""Threshold for excessive LiDAR acquisition problems."""
LIDAR_PROBLEM_THRESHOLD: int = 100

# ============================================================================
# DOMAIN RANDOMIZATION
# ============================================================================
"""Enable domain randomization."""
ENABLE_DOMAIN_RANDOMIZATION: bool = True

"""Range for random delay injection (simulation steps)."""
DOMAIN_RANDOM_DELAY_RANGE: tuple = (0, 5)

"""Enable random LiDAR noise."""
ENABLE_LIDAR_NOISE: bool = True

"""Standard deviation of LiDAR noise (percentage of range)."""
LIDAR_NOISE_STD: float = 0.02

"""Enable random LiDAR dropout."""
ENABLE_LIDAR_DROPOUT: bool = True

"""Probability of LiDAR dropout per measurement."""
LIDAR_DROPOUT_PROB: float = 0.05

"""Range for random steering gain variation."""
DOMAIN_RANDOM_STEERING_GAIN_RANGE: tuple = (0.9, 1.1)

"""Range for random steering bias (degrees)."""
DOMAIN_RANDOM_STEERING_BIAS_RANGE: tuple = (-2.0, 2.0)

"""Range for random P-controller gain variation."""
DOMAIN_RANDOM_KP_RANGE: tuple = (0.6, 1.0)

"""Range for random episode duration variation (steps)."""
DOMAIN_RANDOM_EPISODE_LENGTH_RANGE: tuple = (0.9, 1.1)

"""Range for random vehicle speed variation (multiplier)."""
DOMAIN_RANDOM_SPEED_MULTIPLIER_RANGE: tuple = (0.95, 1.05)

# ============================================================================
# PPO TRAINING HYPERPARAMETERS
# ============================================================================
"""Learning rate for PPO agent."""
PPO_LEARNING_RATE: float = 3e-4

"""Number of environment steps per update."""
PPO_N_STEPS: int = 2048

"""Number of mini-batches."""
PPO_BATCH_SIZE: int = 64

"""Number of epochs for policy update."""
PPO_N_EPOCHS: int = 10

"""Discount factor (gamma)."""
PPO_GAMMA: float = 0.99

"""General Advantage Estimation lambda."""
PPO_GAE_LAMBDA: float = 0.95

"""Clipping range for policy gradient."""
PPO_CLIP_RANGE: float = 0.2

"""Maximum gradient norm."""
PPO_MAX_GRAD_NORM: float = 0.5

"""Verbosity level for PPO training."""
PPO_VERBOSE: int = 1

"""Device for computation ('cpu' or 'cuda')."""
PPO_DEVICE: str = "cpu"

"""TensorBoard log directory."""
TENSORBOARD_LOG_DIR: str = "./tensorboard_logs"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
"""Total timesteps for training."""
TRAINING_TOTAL_TIMESTEPS: int = 200_000

"""Directory for saving trained models."""
MODELS_DIR: str = "./models"

"""Default model checkpoint name."""
DEFAULT_MODEL_NAME: str = "ppo_autonomous_driving"

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
"""Number of evaluation episodes."""
EVAL_NUM_EPISODES: int = 10

"""Use deterministic policy during evaluation."""
EVAL_DETERMINISTIC: bool = True

"""Episode info logging frequency (steps)."""
EVAL_LOG_FREQUENCY: int = 500
