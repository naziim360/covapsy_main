# CoVAPSy Autonomous Driving RL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced reinforcement learning framework for autonomous driving in Webots simulator, featuring PPO-based learning with domain randomization and LiDAR-based perception.

---

## 🎯 Features

- **PPO-Based Learning**: State-of-the-art Proximal Policy Optimization for vehicle control.
- **LiDAR Perception**: Realistic 2D LiDAR simulation with noise and dropout.
- **Domain Randomization**: Robust training with randomized vehicle parameters and sensor characteristics.
- **Multi-Component Reward**: Sophisticated reward function balancing safety, speed, lane centering, and smoothness.
- **P-Controller Speed Management**: Automatic speed regulation based on obstacle distance.
- **Episode Management**: Automatic vehicle repositioning and physics resets.
- **Clean Architecture**: Modular, maintainable codebase with clear separation of concerns.

---

## 🏗️ Repository Architecture & File Structure

The project directory contains the training environment, supervisor tools, and simulation files:

```
covapsy-autonomous-driving/
├── .env                           # Local environment variables (DO NOT COMMIT)
├── .env.example                   # Template for environment configuration
├── .gitignore                     # Git exclusion file
├── LICENSE                        # MIT License
├── README.md                      # Consolidated project documentation
├── requirements.txt               # Python dependencies
├── setup_webots.sh                # Linux/macOS environment setup script
├── setup_webots_windows.bat       # Windows environment setup script
│
├── Raspberry controller/          # Raspberry Pi on-vehicle control module
│   └── Rpi_car_controller.py      # Real-car autonomous control script
│
├── controllers/
│   ├── RL/                        # Reinforcement Learning Module
│   │   ├── config.py              # Centralized hyperparameters & constants
│   │   ├── env.py                 # Gymnasium environment
│   │   ├── lidar.py               # LiDAR sensor management
│   │   ├── reward.py              # Multi-component reward function
│   │   ├── domain_randomization.py # Parameter randomization
│   │   ├── callbacks.py           # Training callbacks
│   │   ├── utils.py               # Utility functions
│   │   ├── train.py               # Training script
│   │   ├── evaluate.py            # Evaluation script
│   │   └── plot_tensorboard.py    # TensorBoard plotting utility
│   │
│   └── supervisor/                # Webots Supervisor Module
│       ├── supervisor.py          # Main supervisor loop
│       ├── spawn.py               # Vehicle positioning
│       ├── checkpoints.py         # Checkpoint tracking
│       └── utils.py               # Supervisor math utilities
│
├── docs/
│   └── architecture.md            # Detailed architecture documentation with diagrams
│
├── protos/                        # Webots custom robot and device definitions
│   ├── TT02_2023b.proto           # Core TT02 vehicle prototype
│   ├── TT02Wheel.proto            # Wheel prototype
│   ├── RpLidarA2.proto            # LiDAR sensor prototype
│   ├── sparringpartner_car_X.proto# Sparring partner vehicles (0, 1, 2)
│   └── ...                        # STL body models and icons
│
├── worlds/                        # Webots world files and track assets
│   ├── Piste_CoVAPSy_2025a.wbt    # Main simulation world
│   └── ...                        # Textures (.jpg) and 3D meshes (.obj)
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.8+
- Webots 2023b or later (download from [cyberbotics.com](https://cyberbotics.com))
- CUDA 11.x (optional, for GPU acceleration)

### 2. Configure Environment Variables
You must configure your Webots environment variables so that Python can locate the Webots controller modules (`vehicle`, `controller`).

- Create a `.env` file by copying the template:
  - **Windows**: `copy .env.example .env`
  - **Linux/macOS**: `cp .env.example .env`
- Open `.env` and set the path to your Webots installation:
  - **Windows Example**:
    ```env
    WEBOTS_HOME=C:\Program Files\Webots
    PYTHONPATH=C:\Program Files\Webots\lib\controller\python
    PATH=C:\Program Files\Webots\lib\controller
    ```
  - **Linux Example**:
    ```env
    WEBOTS_HOME=/usr/local/webots
    PYTHONPATH=/usr/local/webots/lib/controller/python
    PATH=/usr/local/webots/lib/controller
    ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Webots World Setup

For detailed instructions, step-by-step setup guides, and documentation on the simulation environment, please refer to the official resource page:
👉 **[CoVAPSy Webots Simulator Implementation Guide (Eduscol)](https://sti.eduscol.education.fr/si-ens-paris-saclay/ressources_pedagogiques/covapsy-mise-en-oeuvre-du-simulateur-webots)**

### Simulation World Structure
Webots requires both a simulation world file and corresponding custom device/robot definitions. This repository comes pre-packaged with the necessary simulation files:
- **`worlds/`**: Contains the tracks, such as `Piste_CoVAPSy_2025a.wbt` (the main simulator world), alongside textures and meshes.
- **`protos/`**: Contains proto-definitions (like `TT02_2023b.proto` for the vehicle and `RpLidarA2.proto` for the LiDAR) which are dynamically instantiated by the world.

### Configuring Webots for External RL Control
To connect your external Python training script to the Webots environment:
1. Open the Webots application.
2. Load your world file (`worlds/Piste_CoVAPSy_2025a.wbt`).
3. Set the robot's controller to `<extern>` in Webots:
   - In the Scene Tree, locate the robot node (e.g., `DEF TT02_2023b_RL Robot`).
   - Find the `controller` field and set it to `"<extern>"`.
   - Save the world.
4. Ensure the supervisor robot (`DEF Supervisor Supervisor`) is configured to run the supervisor controller:
   - In the Scene Tree, locate the supervisor node.
   - Verify its `controller` field is set to `"supervisor"`.
5. Press the **Pause** button and reset the simulation to time zero before launching the Python script.

---

## 🏋️ Training & Evaluation

The training and evaluation scripts run **inside Webots** as external controllers. They are not standalone Python scripts and require the simulator to be open and running.

### 1. Training
1. Start Webots and load the world (`worlds/Piste_CoVAPSy_2025a.wbt`).
2. Run the training script in your terminal:
   ```bash
   cd controllers/RL
   python train.py --steps 200000 --output ../models/ppo_agent
   ```
3. Press **Play** ▶️ in Webots to begin the simulation.

**Training Options:**
```bash
python train.py --help

Options:
  --model MODEL_PATH      Load existing model to continue training
  --steps STEPS           Total training timesteps (default: 200,000)
  --lr LEARNING_RATE      Override learning rate
  --output PATH           Save model to custom path
```

### 2. Evaluation
1. Run the evaluation script in your terminal:
   ```bash
   cd controllers/RL
   python evaluate.py --model ../models/ppo_agent --episodes 10
   ```
2. Press **Play** ▶️ in Webots to begin.

**Evaluation Options:**
```bash
python evaluate.py --help

Options:
  --model MODEL_PATH      Path to trained model (required)
  --episodes EPISODES     Number of evaluation episodes
  --output PATH           Save results to file
  --stochastic           Use stochastic policy (default: deterministic)
```

---

## 🔧 Configuration Reference

All hyperparameters and tunable variables are centralized in `controllers/RL/config.py`.

### Key Tuning Parameters:

| Category | Hyperparameter | Description | Default Value |
|----------|----------------|-------------|---------------|
| **Vehicle Dynamics** | `VITESSE_MAX_M_S` | Maximum speed of the vehicle | `7.78` |
| | `MAXANGLE_DEGRE` | Maximum steering angle | `16.0` |
| **LiDAR** | `LIDAR_RANGE_MM` | Maximum range of LiDAR sensor | `12000.0` |
| | `LIDAR_SECTOR_SIZE` | Number of measurements / beams | `201` |
| | `SEUIL_COLLISION` | Collision threshold (normalized distance) | (dynamic) |
| **Reward Weights** | `REWARD_WEIGHT_DISTANCE` | Proximity/safety component weight | `1.0` |
| | `REWARD_WEIGHT_SPEED` | Progress/speed component weight | `1.0` |
| | `REWARD_WEIGHT_LANE` | Lane centering weight | `1.0` |
| | `REWARD_WEIGHT_HEADING` | Heading alignment weight | `1.0` |
| **PPO Hyperparams** | `PPO_LEARNING_RATE` | Learning rate | `3e-4` |
| | `PPO_N_STEPS` | Steps per policy update | `2048` |
| | `PPO_BATCH_SIZE` | Mini-batch size | `64` |
| **Domain Rand.** | `ENABLE_DOMAIN_RANDOMIZATION`| Enable vehicle & sensor randomization | `True` |
| | `ENABLE_LIDAR_NOISE` | Add noise to LiDAR data | `True` |
| | `ENABLE_LIDAR_DROPOUT` | Add sensor dropout | `True` |

---

## 📊 Reward Function

The environment utilizes a sophisticated multi-component reward function to balance safety, speed, and comfort:

1. **Distance Component (Safety)**: Logarithmic scaling of the minimum distance to obstacles to discourage driving too close to barriers.
2. **Speed Component (Progress)**: Quadratic function of speed, modulated by the proximity to obstacles (forces slowdowns near curves/barriers).
3. **Lane Centering (Navigation)**: Exponential penalty for asymmetry between the left and right LiDAR readings to direct the vehicle towards the center of the track.
4. **Heading Alignment (Control)**: Symmetric alignment checking in the central LiDAR sector to encourage straight driving.
5. **Steering Smoothness (Comfort)**: Penalizes abrupt angle changes between sequential steps.
6. **Distance Traveled Bonus**: Small bonus per unit of distance covered to encourage forward progression.
7. **Collision Penalty**: High penalty applied immediately when a collision is detected, triggering an episode reset.

---

## 🔄 Code Modules & Responsibilities

The codebase follows a modular design, split between the RL Controller and the Supervisor:

### 1. Reinforcement Learning Controller (`controllers/RL/`)

| Module | Responsibility |
|--------|-----------------|
| `config.py` | Centralized constants, hyperparameters, and environment settings. |
| `env.py` | Core Gymnasium environment interface connecting all sub-modules. |
| `lidar.py` | Handles raw LiDAR data acquisition, cleanup, interpolation, and noise. |
| `reward.py` | Computes the multi-component reward function. |
| `domain_randomization.py` | Implements episode-level domain randomization. |
| `callbacks.py` | Handles model saving, TensorBoard logging, and training callbacks. |
| `utils.py` | Internal utility functions (vehicle math, state representation). |
| `train.py` | Script to set up and train the RL agent. |
| `evaluate.py` | Script to load and evaluate trained policies. |

### 2. Supervisor Controller (`controllers/supervisor/`)

| Module | Responsibility |
|--------|-----------------|
| `supervisor.py` | Main loop controlling Webots supervisor events and communication. |
| `spawn.py` | Handles vehicle spawning, positioning, and orientation randomization. |
| `checkpoints.py` | Manages track checkpoints, lap completion tracking, and out-of-bound detection. |
| `utils.py` | Math utilities for coordinate transformation and distances. |

---

## 🛠️ Troubleshooting

| Issue | Potential Cause | Solution |
|-------|-----------------|----------|
| `ModuleNotFoundError: vehicle` or `controller` | Environment variables in `.env` are missing or incorrect. | Verify that `.env` is configured correctly, then run the activation script (`setup_webots_windows.bat` or `source setup_webots.sh`). |
| `Connection refused` | Webots simulation is not active. | Ensure Webots is open and the simulation is running (press **Play** ▶️). |
| `LiDAR not available` | LiDAR device name mismatch. | Verify the LiDAR name in the world file matches `RpLidarA2` (or adjust `config.py`). |
| `Timeout waiting for supervisor` | Supervisor node is missing or incorrectly set up. | Verify that a `Supervisor` robot node exists in the scene tree and its controller is set to `"supervisor"`. |
---

## 🚗 Real Car Deployment

The script [Rpi_car_controller.py](Raspberry_controller/Rpi_car_controller.py) allows deploying trained policies directly onto the physical TT02 vehicle (running a Raspberry Pi).

### Features
- **RPLidar Support**: Reads scan from the physical USB RPLidar device and formats it exactly like the Webots LiDAR (201 beams, floor values at 301.0 mm).
- **Physical Controls**: Directly controls the steering servo and propulsion motors using hardware PWM modules.
- **Safety Caps & Backups**: Keeps steering limited to 18 degrees and speed capped at 2.0 m/s for safety, with automatic emergency backup and recovery if obstacles come within 250mm of the front or 220mm of the sides.

### Running on the Real Car
1. Copy [Rpi_car_controller.py](Raspberry_controller/Rpi_car_controller.py) and your trained model (e.g. `ppo_agent.zip`) to the vehicle's onboard computer.
2. Create a new virtual environment and install the required dependencies:
   ```bash
   python3 -m venv rl_env
   source rl_env/bin/activate
   pip install rplidar-roboticia rpi-hardware-pwm stable-baselines3 numpy
   ```
3. Run the controller script:
   ```bash
   python3 Rpi_car_controller.py
   ```
4. Follow the interactive menu:
   - Enter `c` to connect hardware (this initializes the RPLidar motor and centers the steering servo).
   - Enter `g` to launch the autonomous agent run.
   - Enter `a` (or press `Ctrl+C`) at any time to perform an emergency stop and disable the PWM signals.

---

## 📄 License & Attribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

- **Last Updated**: 2026
- **Author**: A.NAZIM
