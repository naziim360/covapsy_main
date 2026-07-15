# Architecture Documentation

## System Overview

CoVAPSy is a reinforcement learning framework for autonomous driving in Webots. The system is built around a clean separation of concerns with distinct modules for sensing, decision-making, control, and simulation management.

### High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     WEBOTS SIMULATION                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Vehicle with LiDAR, Motors, Emitter/Receiver            │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                   LiDAR Raw Data
                            ↓
        ┌───────────────────────────────────────┐
        │      LiDAR Sensor Processing           │
        │  • Raw data acquisition                │
        │  • Invalid value filtering             │
        │  • Interpolation (forward/backward)    │
        │  • Noise injection (domain random)     │
        │  • Dropout (domain random)             │
        │  • Normalization to [0,1]              │
        └───────────────────────────┬────────────┘
                                    │
                    Normalized LiDAR [201 values]
                                    ↓
        ┌───────────────────────────────────────┐
        │    Observation Construction            │
        │  • Current LiDAR scan                  │
        │  • Previous LiDAR scan                 │
        │  • Normalized speed                    │
        │  • Previous steering angle             │
        └───────────────────────────┬────────────┘
                                    │
                  Observation Dict (4 channels)
                                    ↓
        ┌───────────────────────────────────────┐
        │      PPO Policy Network                │
        │  • Multi-input policy                  │
        │  • 2 fully connected layers            │
        │  • Deterministic or stochastic output  │
        └───────────────────────────┬────────────┘
                                    │
            Action: Steering [-1, 1]
                                    ↓
        ┌───────────────────────────────────────┐
        │   Domain Randomization Layer           │
        │  • Steering gain variation             │
        │  • Steering bias injection             │
        │  • P-controller gain randomization     │
        │  • Speed scaling randomization         │
        └───────────────────────────┬────────────┘
                                    │
        Randomized Steering Command
                                    ↓
        ┌───────────────────────────────────────┐
        │  Vehicle Command Application           │
        │  • Steering angle (→ vehicle motor)    │
        │  • P-based speed control               │
        │  • Obstacle distance based regulation  │
        └───────────────────────────┬────────────┘
                                    │
                    Steering & Speed Commands
                                    ↓
┌──────────────────────────────────────────────────────────────────┐
│                     WEBOTS SIMULATION STEP                        │
│  • Physics update                                                │
│  • Vehicle dynamics                                              │
│  • Collision detection                                           │
│  • New LiDAR measurement                                         │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                   New LiDAR Data
                            ↓
        ┌───────────────────────────────────────┐
        │  Multi-Component Reward Computation    │
        │  1. Distance reward (obstacle safety)  │
        │  2. Speed reward (forward progress)    │
        │  3. Lane centering reward              │
        │  4. Heading alignment reward           │
        │  5. Steering smoothness penalty        │
        │  6. Distance traveled bonus            │
        │  7. Collision penalty (if applicable)  │
        └───────────────────────────┬────────────┘
                                    │
                    Reward + Episode Status
                                    ↓
        ┌───────────────────────────────────────┐
        │     PPO Training Update                │
        │  • Compute advantage                   │
        │  • Policy gradient step                │
        │  • Value function update               │
        │  • Entropy regularization              │
        └───────────────────────────────────────┘
```

## Module Architecture

### RL Agent Modules (`controllers/RL/`)

#### 1. **config.py**
Central configuration repository.

```python
# Vehicle Parameters
VITESSE_MAX_M_S = 28.0 / 3.6  # 7.78 m/s
MAXANGLE_DEGRE = 16.0

# LiDAR Configuration
LIDAR_RANGE_MM = 12000
LIDAR_SECTOR_SIZE = 201
SEUIL_COLLISION = 200.0 / LIDAR_RANGE_MM

# Reward Weights
REWARD_WEIGHT_DISTANCE = 1.0
REWARD_WEIGHT_SPEED = 3.0
REWARD_WEIGHT_LANE = 3.0
REWARD_WEIGHT_HEADING = 1.5
REWARD_WEIGHT_ANGLE_PENALTY = 2.0

# Domain Randomization
ENABLE_DOMAIN_RANDOMIZATION = True
ENABLE_LIDAR_NOISE = True
ENABLE_LIDAR_DROPOUT = True

# PPO Hyperparameters
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
```

**Responsibility**: Single source of truth for all constants.

#### 2. **env.py**
Gymnasium environment integrating all components.

```
class WebotsGymEnvironment(Driver, gym.Env):
    - Inherits from Webots Driver for vehicle control
    - Implements Gymnasium interface
    - Coordinates:
        * LiDAR acquisition (LidarManager)
        * Observation building
        * Reward computation (RewardManager)
        * Domain randomization (DomainRandomizer)
        * Speed control (SpeedController)
    - Handles episode resets and crash recovery
    - Communicates with Webots supervisor
```

**Key Methods**:
- `step(action)`: Execute one environment step
- `reset()`: Initialize new episode
- `_get_observation()`: Acquire and process sensor data
- `_compute_reward_and_termination()`: Evaluate performance
- `_handle_crash_reset()`: Communicate with supervisor

#### 3. **lidar.py**
LiDAR sensor data acquisition and preprocessing.

```
class LidarManager:
    # Data acquisition pipeline:
    1. get_raw_lidar_mm()
       └─ Webots LiDAR (361 values, -180° to +180°)
       └─ Extract sector [-100°, +100°] (201 values)
    
    2. _clean_invalid_values()
       └─ Replace 0 and inf with NaN
    
    3. _interpolate_missing_values()
       └─ Forward fill + backward fill
    
    4. _apply_noise() [optional]
       └─ Add Gaussian noise (domain randomization)
    
    5. _apply_dropout() [optional]
       └─ Random measurement dropout (robustness)
    
    6. Normalization to [0, 1]
       └─ Divide by LIDAR_RANGE_MM
       └─ Clip to valid range
    
    Result: Normalized LiDAR [201,] float32 array
```

**Measurement Flow**:
```
Raw Webots Range Image (361 values)
         ↓
  Sector Extraction (201 values)
         ↓
  Unit Conversion (mm)
         ↓
  Invalid Value Cleaning (NaN)
         ↓
  Interpolation (forward/backward fill)
         ↓
  Optional: Noise Addition
         ↓
  Optional: Dropout Application
         ↓
  Normalization [0, 1]
         ↓
  Clipping + Conversion to float32
```

#### 4. **reward.py**
Multi-component reward function.

```
class RewardManager:
    compute_reward(observation, angle, timestep) → (reward, done)
    
    Components:
    1. _compute_distance_reward()
       └─ Log-scale of minimum distance
       └─ W_dist = 1.0
    
    2. _compute_speed_reward()
       └─ Quadratic speed * safety_factor
       └─ W_speed = 3.0
    
    3. _compute_lane_centering_reward()
       └─ Exponential penalty for left/right asymmetry
       └─ W_lane = 3.0
    
    4. _compute_heading_reward()
       └─ Exponential penalty for heading misalignment
       └─ W_heading = 1.5
    
    5. _compute_steering_smoothness_penalty()
       └─ Penalty for abrupt angle changes
       └─ W_angle = -2.0
    
    6. _compute_distance_traveled_bonus()
       └─ Small bonus per meter
       └─ W_travel = 0.1
    
    Final: R = Σ weighted_components
```

**Reward Formula**:
```
R = 1.0  × log(min_dist × 120 + ε)
  + 3.0  × (3*v - v²) × safety_factor
  + 3.0  × exp(-(center_error / 0.1)²)
  + 1.5  × exp(-(heading_error / 0.1)²)
  - 2.0  × |Δangle| / (2*MAXANGLE)
  + 0.1  × distance_traveled

  collision: R = -100.0
```

#### 5. **domain_randomization.py**
Episode-level parameter randomization for robust training.

```
class DomainRandomizer:
    randomize_episode_parameters()
    ├─ Episode length multiplier: [0.9, 1.1]
    ├─ Steering gain multiplier: [0.9, 1.1]
    ├─ Steering bias: [-2.0°, 2.0°]
    ├─ P-controller gain multiplier: [0.6, 1.0]
    ├─ Speed multiplier: [0.95, 1.05]
    └─ LiDAR delay: [0, 5] steps

    Applies:
    ├─ Steering gain to action
    ├─ Speed scaling to command
    ├─ Controller gain modification
    └─ LiDAR sensor delay
```

**Purpose**: Improve generalization and robustness to parameter variations.

#### 6. **utils.py**
Utility classes and functions.

```
class SpeedController:
    - Proportional control based on obstacle distance
    - Maintains target safety distance
    - Speed = v_min + kp * (d_min - d_target)

class VehicleController:
    - Apply steering commands (deg → rad)
    - Apply speed commands (m/s → km/h)

class ObservationBuilder:
    - Construct Dict observation
    - Handle initial vs. ongoing observations
    - Extract front LiDAR sector

class AngleUtils:
    - Normalize angles
    - Compute angle differences
    - Clamp angles

Helper functions:
    - create_info_dict()
    - clip_value()
```

#### 7. **callbacks.py**
Training monitoring and management.

```
Classes:
- TrainingProgressCallback: Custom Stable-Baselines3 callback to monitor and log training progress (average reward, episode length, maximum reward) at a defined log frequency and compute statistics.
```

#### 8. **train.py**
Training script.

```
Main flow:
1. Create environment
2. Verify Gymnasium compatibility
3. Create or load PPO model
4. Train for N timesteps
5. Save trained model
6. Print summary statistics

Features:
- Command-line arguments for flexible training
- Callback integration
- Model checkpoint support
- Learning rate override
```

#### 9. **evaluate.py**
Evaluation script.

```
Main flow:
1. Load trained model
2. Disable domain randomization
3. Run N episodes
4. Collect statistics
5. Display summary and save results

Metrics:
- Mean/std/min/max reward
- Episode length
- Distance traveled
- Crash rate
```

### Supervisor Modules (`controllers/supervisor/`)

#### 1. **supervisor.py**
Main supervisor loop.

```
Responsibilities:
1. Vehicle Safety Monitoring
   └─ Check position bounds
   └─ Auto-reposition if out of bounds

2. Message Processing
   └─ Receive "crash" messages from agent
   └─ Trigger vehicle reset sequence
   └─ Send acknowledgment

3. Vehicle Reset
   └─ Call SpawnManager for new positions
   └─ Reset physics
   └─ Stabilize simulation

Main Loop:
while simulation_running:
    1. Check vehicle position validity
    2. Process incoming messages
    3. Reset vehicles if requested
    4. Send acknowledgment
```

#### 2. **spawn.py**
Vehicle positioning and randomization.

```
class SpawnManager:
    - Select random distinct spawn points
    - Randomize direction (normal/reversed)
    - Generate random spawn coordinates with noise
    - Normalize angles to [-π, π]

Spawn Points:
- Training track: 4 positions
- Test track: 8 positions

Randomization:
- X, Y: Uniform within defined ranges
- Angle: Base ± π/12 noise
- Direction: 0 (forward) or 1 (reversed)

function get_spawn_point(index, direction, angle_noise):
    x = uniform(range_x[0], range_x[1])
    y = uniform(range_y[0], range_y[1])
    angle = base_angle ± angle_noise
    if direction == 1:
        angle += π
    return normalize(angle)
```

#### 3. **checkpoints.py**
Milestone and lap tracking.

```
class Checkpoint:
    - Define circular detection zones
    - Check point containment
    - Calculate distance to checkpoint

class CheckpointTracker:
    - Track vehicle progress through circuit
    - Count completed laps
    - Record best lap time
    - Monitor lap progress percentage

Methods:
- check_progress(x, y, time): Update on each vehicle position
- reset(): Initialize new episode
- get_progress(): Return lap completion fraction
- get_lap_count(): Return completed laps
- get_best_lap_time(): Return best lap time
```

#### 4. **utils.py**
Supervisor utility functions.

```
Functions:
- normalize_angle(angle): Convert to [-π, π]
- clip_value(value, low, high): Clamp to range
- is_position_valid(x, y, z): Check bounds
- get_safe_position(): Return fallback position
- distance_between_points(): Euclidean distance
- create_rotation_array(angle): Webots rotation format
```

## Data Structures

### Observation Dictionary
```python
{
    "current_lidar": np.array([201,], dtype=float32),  # Current scan [0,1]
    "previous_lidar": np.array([201,], dtype=float32), # Previous scan [0,1]
    "current_speed": np.array([1,], dtype=float32),    # Normalized [0,1]
    "previous_angle": np.array([1,], dtype=float32),   # Normalized [-1,1]
}
```

### Action Space
```python
Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)
# Maps to steering angle:
steering_angle_deg = action[0] * MAXANGLE_DEGRE
```

### Info Dictionary
```python
{
    "vitesse_ms": float,        # Current speed
    "angle_degre": float,       # Current steering angle
    "distance_min_m": float,    # Min distance to obstacle
    "total_distance": float,    # Distance traveled in episode
    "episode_step": int,        # Current step number
    "crash_count": int,         # Cumulative crashes
}
```

## Episode Lifecycle

### Training Episode Flow

```
reset() called
    ↓
[Initialize]
    - Reset vehicle state
    - Stabilize simulation (20 steps)
    - Handle crash recovery if needed
    - Randomize domain parameters
    - Clear managers
    ↓
Initial observation acquired
    ↓
episode_loop: while not done:
    ├─ step(action) called
    ├─ [Action Processing]
    │  ├─ Parse steering action
    │  ├─ Apply domain randomization
    │  └─ Compute speed via P-controller
    ├─ [Command Application]
    │  ├─ Apply steering
    │  ├─ Apply speed
    │  └─ Webots step()
    ├─ [Observation & Reward]
    │  ├─ Get new LiDAR data
    │  ├─ Build observation
    │  ├─ Compute multi-component reward
    │  └─ Check termination
    ├─ Update state for next iteration
    └─ Return (obs, reward, done, truncated, info)
        ↓
    if done:
        └─ reset() called for next episode

Termination Conditions:
    1. Episode length exceeded (RESET_STEP)
    2. Excessive LiDAR problems
    3. Training loop end
```

### Crash Recovery Flow

```
During step():
    Collision detected (distance < threshold)
        ↓
    Reward = -100.0
    done = False (episode continues)
    crash_count += 1
        ↓
During next reset():
    if crash_count > 0:
        ├─ Stop vehicle
        ├─ Send "crash" message to supervisor
        ├─ Wait for acknowledgment (timeout 1000 steps)
        ├─ Supervisor receives message
        ├─ Supervisor resets physics
        ├─ Supervisor repositions vehicle
        ├─ Supervisor sends acknowledgment
        ├─ Agent receives acknowledgment
        ├─ Agent resets episode state
        └─ Episode continues from new position
```

## Communication Protocol

### Agent → Supervisor
```
Message Format: "voiture crash <packet_number>"
Channel: Emitter (string)
Trigger: Collision detected or recovery needed
Purpose: Request vehicle repositioning
```

### Supervisor → Agent
```
Message Format: "voiture replacee num : <packet_number>"
Channel: Receiver (string)
Trigger: After vehicle reset completion
Purpose: Acknowledge successful repositioning
```

## Performance Optimization

### LiDAR Processing
- Pre-compute forward/backward fill indices
- Vectorized NumPy operations for all processing
- Minimal memory allocation per step

### Reward Computation
- Vectorized array operations
- Pre-allocated observation dictionary
- Lazy evaluation of reward components

### Training
- Batch processing via PPO (N_STEPS = 2048)
- Vectorized environment stepping


---

**Last Updated**: 2026