"""
Real Car Controller for TT02 vehicle - aligned with the RL Domain Randomization training environment.

This controller:
1. Acquires raw LiDAR scans from RPLidar, maps them to the [-100°, +100°] sector (201 beams).
2. Builds observation dictionary matching the model input shape and normalization.
3. Normalizes speed by training VITESSE_MAX_M_S, and previous angle by training MAXANGLE_DEGRE.
4. Predicts steering actions using the trained SB3 PPO model.
5. Regulates vehicle speed with a safe, hardware-limited P-controller.
6. Automatically detects collisions and triggers emergency backup logic.
"""

from rplidar import RPLidar
import time
import numpy as np
from rpi_hardware_pwm import HardwarePWM
import threading
from collections import deque
from stable_baselines3 import PPO

# ============================================================================
# DYNAMIC CONFIGURATION IMPORT
# ============================================================================
try:
    from controllers.RL import config as training_config
    TRAINING_VITESSE_MAX_M_S = training_config.VITESSE_MAX_M_S
    TRAINING_MAXANGLE_DEGRE = training_config.MAXANGLE_DEGRE
    TRAINING_LIDAR_SECTOR_SIZE = training_config.LIDAR_SECTOR_SIZE
    TRAINING_LIDAR_MAX_MM = training_config.LIDAR_RANGE_MM
    print("[INFO] Successfully imported training config. Auto-syncing normalization constants:")
    print(f"       VITESSE_MAX_M_S = {TRAINING_VITESSE_MAX_M_S} m/s")
    print(f"       MAXANGLE_DEGRE  = {TRAINING_MAXANGLE_DEGRE}°")
except ImportError:
    # Standalone Raspberry Pi fallback values (sync with controllers/RL/config.py)
    TRAINING_VITESSE_MAX_M_S = 7.78
    TRAINING_MAXANGLE_DEGRE = 16.0
    TRAINING_LIDAR_SECTOR_SIZE = 201
    TRAINING_LIDAR_MAX_MM = 12000.0
    print("[WARNING] Could not import controllers.RL.config. Using default training fallback values:")
    print(f"          VITESSE_MAX_M_S = {TRAINING_VITESSE_MAX_M_S} m/s")
    print(f"          MAXANGLE_DEGRE  = {TRAINING_MAXANGLE_DEGRE}°")

# ============================================================================
# PHYSICAL HARDWARE CONSTANTS
# ============================================================================

# --- Physical Speed & Speed Controller Configuration ---
VITESSE_MAX_M_S       = 3.0      # Target speed command limit for the real car
VITESSE_MIN_M_S       = 0.11     # Floor speed limit
VITESSE_SECURITE_MIN  = 0.11     # Minimum speed when close to obstacles
KP_VITESSE            = 0.4      # Proportional speed controller gain
KD_VITESSE            = 0.0      # Derivative speed controller gain (unused/0.0)
DISTANCE_CIBLE_M      = 0.7      # Target safety distance from obstacles (meters)

# --- Physical Steering Configuration ---
MAXANGLE_DEGRE        = 18.0     # Physical steering range in degrees (limit of real servo)

# --- Collision Recovery (Backup) Thresholds ---
SEUIL_COLLISION_FRONT_MM = 250.0  # Trigger reverse if front obstacle closer than 250mm
SEUIL_COLLISION_SIDE_MM  = 220.0  # Trigger reverse if side obstacle closer than 220mm

# --- Sector Indices (Matching Webots [-100°, +100°] sector) ---
FRONT_IDX_START = 55
FRONT_IDX_END   = 146

# --- Propulsion PWM Signal Mapping ---
PWM_STOP_PROP        = 7.5      # PWM duty cycle (%) for stopping
POINT_MORT_PROP      = 0.4      # Dead band offset
DELTA_PWM_MAX_PROP   = 1.5      # Max PWM dynamic range
VITESSE_MAX_M_S_HARD = 8.0      # Absolute hardware max speed
VITESSE_MAX_M_S_SOFT = 2.0      # Safety cap for physical testing
VITESSE_MAX_AR_M_S   = 4.0      # Reverse speed limit

# --- Steering Servo PWM Signal Mapping ---
ANGLE_PWM_MIN        = 6.3      # Minimum servo duty cycle (%)
ANGLE_PWM_MAX        = 9.5      # Maximum servo duty cycle (%)
ANGLE_DIFF           = ANGLE_PWM_MAX - ANGLE_PWM_MIN
ANGLE_PWM_CENTRE     = 7.4      # Center servo duty cycle (%)
ANGLE_DEGRE_MAX_HARD = MAXANGLE_DEGRE

# --- RPLidar Device Setup ---
LIDAR_PORT      = '/dev/ttyUSB0'
LIDAR_BAUDRATE  = 256000

# ============================================================================
# CHASSIS CONTROL INTERFACE
# ============================================================================

class Chassis:
    """
    Interfaces directly with the propulsion and steering servo hardware via PWM.
    """
    def __init__(self):
        self.pwm_prop = HardwarePWM(pwm_channel=0, hz=50)
        self.pwm_dir  = HardwarePWM(pwm_channel=1, hz=50)
        self.vitesse_consigne   = 0.0
        self.direction_consigne = 0.0

    def demarrage_voiture(self):
        """Initialize PWM signals for starting the vehicle."""
        print("[CHASSIS] Starting vehicle hardware...")
        self.pwm_prop.start(PWM_STOP_PROP)
        self.pwm_dir.start(ANGLE_PWM_CENTRE)

    def arret_voiture(self):
        """Stop the vehicle and shut down PWM signals."""
        self.set_vitesse_m_s(0.0)
        self.pwm_prop.stop()
        self.pwm_dir.stop()
        print("[CHASSIS] Vehicle hardware stopped.")

    def get_vitesse(self) -> float:
        """Get the current commanded speed in m/s."""
        return self.vitesse_consigne

    def get_direction(self) -> float:
        """Get the current commanded steering angle in degrees."""
        return self.direction_consigne

    def set_vitesse_m_s(self, vitesse_m_s: float):
        """
        Convert velocity command to corresponding PWM duty cycle and apply to motors.
        """
        vitesse_m_s = float(vitesse_m_s)
        # Limit speed command to safe test ranges
        vitesse_m_s = np.clip(vitesse_m_s, -VITESSE_MAX_AR_M_S, VITESSE_MAX_M_S_SOFT)

        if vitesse_m_s == 0.0:
            self.pwm_prop.change_duty_cycle(PWM_STOP_PROP)
        elif vitesse_m_s > 0.0:
            delta = vitesse_m_s * DELTA_PWM_MAX_PROP / VITESSE_MAX_M_S_HARD
            self.pwm_prop.change_duty_cycle(PWM_STOP_PROP + POINT_MORT_PROP + delta)
        else:
            delta = -vitesse_m_s * DELTA_PWM_MAX_PROP / VITESSE_MAX_M_S_HARD
            self.pwm_prop.change_duty_cycle(PWM_STOP_PROP - POINT_MORT_PROP - delta)

        self.vitesse_consigne = vitesse_m_s

    def set_direction_degre(self, angle_degre: float):
        """
        Convert steering angle (degrees) to corresponding PWM duty cycle and apply to servo.
        """
        angle_degre = float(np.clip(angle_degre, -ANGLE_DEGRE_MAX_HARD, ANGLE_DEGRE_MAX_HARD))
        angle_pwm   = ANGLE_PWM_CENTRE - ANGLE_DIFF * angle_degre / (2 * ANGLE_DEGRE_MAX_HARD)
        angle_pwm   = float(np.clip(angle_pwm, ANGLE_PWM_MIN, ANGLE_PWM_MAX))
        self.pwm_dir.change_duty_cycle(angle_pwm)
        self.direction_consigne = angle_degre

    def recule(self, right_side_mm: np.ndarray, left_side_mm: np.ndarray,
               vitesse_m_s: float = 1.5, duree_s: float = 0.5):
        """
        Trigger emergency recovery reverse sequence.
        Steers away from the closest obstacles while backing up.
        """
        sign = np.sign(np.mean(right_side_mm - left_side_mm))
        self.set_vitesse_m_s(-vitesse_m_s)
        self.set_direction_degre(MAXANGLE_DEGRE * sign)
        time.sleep(duree_s)
        self.set_vitesse_m_s(0.0)
        time.sleep(0.2)


# ============================================================================
# SENSOR SENSOR MANAGER
# ============================================================================

class Lidar_TT02:
    """
    Manages background RPLidar sensor readings and parses the target [-100°, +100°] sector.
    """
    def __init__(self):
        self.lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
        self.acqui_lidar       = np.zeros(360, dtype=np.float32)
        self.acqui_lidar_ready = np.zeros(TRAINING_LIDAR_SECTOR_SIZE, dtype=np.float32)
        self.drapeau_nouveau_scan = False
        self.Run_lidar            = False
        self._lock                = threading.Lock()

    def demarrage_lidar(self):
        """Initialize connection and start RPLidar motor."""
        self.lidar.connect()
        self.lidar.start_motor()
        time.sleep(2)
        print("[LIDAR] Lidar connected and motor started.")

    def arret_lidar(self):
        """Stop RPLidar motor and close connection."""
        self.lidar.stop_motor()
        self.lidar.stop()
        time.sleep(1)
        self.lidar.disconnect()
        print("[LIDAR] Lidar stopped and disconnected.")

    def lidar_scan(self):
        """
        Background thread reading scans continuously.
        """
        while self.Run_lidar:
            try:
                for scan in self.lidar.iter_scans(scan_type='express'):
                    for _, angle_lidar, distance in scan:
                        angle = int(angle_lidar) % 360
                        with self._lock:
                            self.acqui_lidar[angle] = distance

                    self.acqui_lidar_ready = self._build_sector()

                    with self._lock:
                        self.drapeau_nouveau_scan = True

                    if not self.Run_lidar:
                        break

            except Exception as e:
                print(f"[LIDAR ERROR] Data acquisition issue: {e}")

    def _build_sector(self) -> np.ndarray:
        """
        Extracts the frontal [-100°, +100°] sector (201 measurements).
        Maps 360° RPLidar indices to match simulated Webots LiDAR format:
        - Indices [260:360] map to left sector indices [0:99].
        - Indices [0:101] map to center and right sector indices [100:200].
        Invalid/zero readings are replaced with 301.0 mm (identical to training floor value).
        """
        with self._lock:
            raw = self.acqui_lidar.copy()

        sector = np.empty(TRAINING_LIDAR_SECTOR_SIZE, dtype=np.float32)
        sector[:100] = raw[260:360]
        sector[100:] = raw[0:101]

        # Replace invalid values (0.0 mm) with fallback floor (301.0 mm)
        sector = np.where(sector == 0.0, 301.0, sector)

        return sector.astype(np.float32)

    def get_drapeau(self) -> bool:
        """Check if a new scan is ready."""
        with self._lock:
            return self.drapeau_nouveau_scan

    def get_run(self) -> bool:
        """Check if the scan thread is running."""
        return self.Run_lidar

    def set_run(self, value: bool):
        """Set execution flag."""
        self.Run_lidar = value


# ============================================================================
# SPEED PROPORTIONAL-DERIVATIVE CONTROLLER
# ============================================================================

def PD_correction_speed(last_error: float, min_distance_normalized: float):
    """
    Computes commanded velocity using the proportional speed controller logic.
    Inputs are normalized based on training constraints to align policy inputs.
    """
    min_distance_m = min_distance_normalized * TRAINING_LIDAR_MAX_MM / 1000.0
    error = min_distance_m - DISTANCE_CIBLE_M
    vitesse_commande = (
        VITESSE_SECURITE_MIN
        + KP_VITESSE * error
        + KD_VITESSE * (error - last_error)
    )
    vitesse_commande = float(np.clip(vitesse_commande, VITESSE_MIN_M_S, VITESSE_MAX_M_S))
    return vitesse_commande, error


# ============================================================================
# AUTONOMOUS DRIVING LOOP
# ============================================================================

def conduite(lidar: Lidar_TT02, voiture: Chassis, model):
    """
    Executes real-time autonomous driving logic:
    1. Collects and pre-processes RPLidar readings.
    2. Runs collision checking and backup maneuvers.
    3. Builds the normalization vector (matching config.py).
    4. Predicts action with policy network.
    5. Applies commands to the chassis actuators.
    """
    print("[DRIVE] Starting autonomous driving mode...")

    # Wait for first valid scan
    while lidar.get_run() and not lidar.get_drapeau():
        time.sleep(0.005)

    # Initialize observation histories
    first_scan_mm  = lidar.acqui_lidar_ready.copy()
    current_lidar  = np.clip(first_scan_mm / TRAINING_LIDAR_MAX_MM, 0.0, 1.0).astype(np.float32)
    previous_lidar = current_lidar.copy()
    previous_angle = np.array([0.0], dtype=np.float32)
    last_error     = 0.0

    while lidar.get_run():
        # Block until next frame scan
        while lidar.get_run() and not lidar.get_drapeau():
            time.sleep(0.001)

        if not lidar.get_run():
            break

        lidar_mm = lidar.acqui_lidar_ready.copy()

        with lidar._lock:
            lidar.drapeau_nouveau_scan = False

        # --- Collision Detection ---
        front_mm = lidar_mm[FRONT_IDX_START:FRONT_IDX_END]
        mini_front = float(np.min(front_mm))

        sides_mm = np.concatenate([lidar_mm[:FRONT_IDX_START], lidar_mm[FRONT_IDX_END:]])
        mini_side = float(np.min(sides_mm))

        if mini_front < SEUIL_COLLISION_FRONT_MM or mini_side < SEUIL_COLLISION_SIDE_MM:
            print(f"[RECOVERY] Collision Alert! Front: {mini_front:.1f}mm, Side: {mini_side:.1f}mm. Reversing...")
            voiture.recule(
                right_side_mm=lidar_mm[FRONT_IDX_END:],
                left_side_mm=lidar_mm[:FRONT_IDX_START]
            )
            # Reset histories after recovery
            current_lidar = np.clip(lidar.acqui_lidar_ready.copy() / TRAINING_LIDAR_MAX_MM, 0.0, 1.0).astype(np.float32)
            previous_lidar = current_lidar.copy()
            previous_angle = np.array([0.0], dtype=np.float32)
            last_error = 0.0
            continue

        # --- Normalize Observation Elements (Synchronized with Simulation Config) ---
        current_lidar = np.clip(lidar_mm / TRAINING_LIDAR_MAX_MM, 0.0, 1.0).astype(np.float32)
        current_speed_norm = float(np.clip(voiture.get_vitesse() / TRAINING_VITESSE_MAX_M_S, 0.0, 1.0))

        obs = {
            "current_lidar":  current_lidar,
            "previous_lidar": previous_lidar,
            "current_speed":  np.array([current_speed_norm], dtype=np.float32),
            "previous_angle": previous_angle,
        }

        # --- Model Inference ---
        action, _ = model.predict(obs, deterministic=True)
        angle_norm = float(action[0])  # Output range [-1, 1]

        # Convert normalized action [-1, 1] to target physical steering angle (degrees)
        angle_deg = float(np.clip(
            angle_norm * TRAINING_MAXANGLE_DEGRE,
            -ANGLE_DEGRE_MAX_HARD,
            ANGLE_DEGRE_MAX_HARD
        ))

        # Calculate target speed command based on frontal obstacle clearance
        front_narrow_norm = current_lidar[90:110]
        min_dist_norm     = float(np.min(front_narrow_norm))
        speed_ms, last_error = PD_correction_speed(last_error, min_dist_norm)

        # Apply physical commands
        voiture.set_direction_degre(angle_deg)
        voiture.set_vitesse_m_s(speed_ms)

        # Update historical trackers for next loop
        previous_lidar = current_lidar.copy()
        previous_angle = np.array(
            [voiture.get_direction() / TRAINING_MAXANGLE_DEGRE],
            dtype=np.float32
        )

    print("[DRIVE] Autonomous driving loop ended.")
    voiture.set_vitesse_m_s(0.0)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 70)
    print("CoVAPSy TT02 Real-Car Autonomous Driver")
    print(f"  Speed target limits: min {VITESSE_MIN_M_S} m/s, max {VITESSE_MAX_M_S} m/s")
    print(f"  Physical steering range: ±{MAXANGLE_DEGRE}° (Safety cap: {ANGLE_DEGRE_MAX_HARD}°)")
    print("=" * 70)

    # --- Load Trained RL Policy ---
    # Attempts loading model path
    MODEL_PATH = "ppo_agent.zip"
    try:
        model = PPO.load(MODEL_PATH)
        print(f"[MODEL] Policy successfully loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Fichier modèle introuvable : {MODEL_PATH}")
        return

    lidar   = Lidar_TT02()
    voiture = Chassis()

    system_active            = False
    thread_scan_lidar        = None
    thread_conduite_autonome = None

    while True:
        choice = input("\n'c' = Connect hardware | 'q' = Quit: ").strip().lower()

        if choice == 'q':
            break

        if choice == 'c':
            try:
                lidar.demarrage_lidar()
                lidar.set_run(True)
                voiture.demarrage_voiture()
                time.sleep(1)

                thread_scan_lidar = threading.Thread(
                    target=lidar.lidar_scan, daemon=True
                )
                thread_scan_lidar.start()
                time.sleep(1)

                system_active = True
                print("[SYSTEM] Hardware connected and active.")
            except Exception as hardware_err:
                print(f"[SYSTEM ERROR] Failed to connect hardware: {hardware_err}")
                continue

        if not system_active:
            print("[SYSTEM] Connect hardware first using 'c'.")
            continue

        choice = input("'g' = Go (Launch autonomous run): ").strip().lower()
        if choice != 'g':
            continue

        if not lidar.get_run():
            print("[SYSTEM] LiDAR is not active. Please reconnect with 'c'.")
            continue

        thread_conduite_autonome = threading.Thread(
            target=conduite,
            args=(lidar, voiture, model),
            daemon=True,
        )
        thread_conduite_autonome.start()

        # Run interface controller
        while True:
            try:
                stop_choice = input("'a' = Stop execution\n").strip().lower()
                if stop_choice == 'a':
                    voiture.set_vitesse_m_s(0.0)
                    lidar.set_run(False)
                    break
            except KeyboardInterrupt:
                print("\n[SYSTEM] Emergency manual override triggered.")
                voiture.set_vitesse_m_s(0.0)
                lidar.set_run(False)
                break

        # Safely shut down background processes
        if thread_scan_lidar and thread_scan_lidar.is_alive():
            thread_scan_lidar.join(timeout=3)
        if thread_conduite_autonome and thread_conduite_autonome.is_alive():
            thread_conduite_autonome.join(timeout=3)

        lidar.arret_lidar()
        voiture.arret_voiture()
        system_active = False


if __name__ == "__main__":
    main()