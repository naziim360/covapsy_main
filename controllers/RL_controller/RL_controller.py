"""
Environnement Gymnasium pour véhicule autonome Webots
Version refactorisée avec :
- Action unique : angle de braquage
- Contrôle de vitesse par correcteur P basé sur la distance minimale
"""


import time
import numpy as np
import gymnasium as gym
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from controller import Lidar  # type: ignore
from vehicle import Driver  # type: ignore

# ============================================================================
# CONSTANTES
# ============================================================================
VITESSE_MAX_M_S = 28.0 / 3.6  # 28 km/h → 7.78 m/s
VITESSE_MIN_M_S = 0.2  # Vitesse minimale pour éviter l'arrêt complet
MAXANGLE_DEGRE = 16.0  # Angle de braquage maximal (degrés)

LIDAR_RANGE_MM = 12000  # Portée maximale du lidar (12m)
LIDAR_SECTOR_SIZE = 201  # Nombre de mesures (-100° à +100°)

SEUIL_COLLISION = 200.0/ LIDAR_RANGE_MM  # (entrainement) 
#SEUIL_COLLISION = 175.0/ LIDAR_RANGE_MM  # 175mm normalisé (test)


RECEIVER_SAMPLING_PERIOD = 64  # ms
RESET_STEP = 2000  # Nombre de pas avant fin d'épisode
MAX_LIDAR_RETRY = 50  # Tentatives max pour acquisition lidar

# Paramètres du correcteur P de vitesse
KP_VITESSE = 0.8  # Gain proportionnel
DISTANCE_CIBLE_M = 0.8  # Distance de sécurité cible
VITESSE_SECURITE_MIN = 0.4  # Vitesse min quand proche obstacle (m/s)


class WebotsGymEnvironment(Driver, gym.Env):
    """Environnement Gym pour l'apprentissage d'un véhicule autonome"""
    
    def __init__(self):
        super().__init__()

        # État du véhicule
        self.consigne_angle = 0.0  # Angle de braquage (degrés)
        self.last_angle = 0.0  
        self.consigne_vitesse = VITESSE_MIN_M_S  # Vitesse (m/s)
        
        # Métriques de supervision
        self.numero_crash = 0
        self.nb_pb_lidar = 0
        self.nb_pb_acqui_lidar = 0
        self.reset_counter = 0
        self.packet_number = 0
        self.total_distance = 0.0  # Distance parcourue dans l'épisode

        # Dispositifs Webots
        self.emitter = super().getDevice("emitter")
        self.receiver = super().getDevice("receiver")
        self.receiver.enable(RECEIVER_SAMPLING_PERIOD)

        # Lidar
        self.lidar = super().getDevice("RpLidarA2")
        timestep = int(super().getBasicTimeStep())
        self.lidar.enable(timestep)
        self.lidar.enablePointCloud()
        
        # Temps de simulation (pour calcul de distance)
        self.timestep_s = timestep / 1000.0

        # ====================================================================
        # SPACES GYMNASIUM
        # ====================================================================
        
        # Action space : angle de braquage uniquement [-1, 1]
        # Sera mappé sur [-MAXANGLE_DEGRE, +MAXANGLE_DEGRE]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        self.observation_space = gym.spaces.Dict({
            "current_lidar": gym.spaces.Box(
                low=0.0, high=1.0, shape=(LIDAR_SECTOR_SIZE,), dtype=np.float32
            ),
            "previous_lidar": gym.spaces.Box(
                low=0.0, high=1.0, shape=(LIDAR_SECTOR_SIZE,), dtype=np.float32
            ),
            "current_speed": gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "previous_angle": gym.spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })

        # Observation initiale
        self.observation = None

    # ========================================================================
    # ACQUISITION LIDAR
    # ========================================================================
    
    def get_lidar_mm(self):
        """
        Récupère les données lidar et les nettoie.
        Retourne un array de 201 valeurs en millimètres.
        """
        raw = np.asarray(self.lidar.getRangeImage(), dtype=np.float64)

        # Extraction du secteur [-100° à +100°] → 201 valeurs
        sector = raw[80:281]

        # Remplacer valeurs invalides (0 ou inf) par NaN
        sector = np.where((sector == 0) | np.isinf(sector), np.nan, sector)

        # Forward fill (propagation des valeurs valides vers l'avant)
        left = sector.copy()
        mask = np.isnan(left)
        idx = np.where(~mask, np.arange(len(left)), 0)
        np.maximum.accumulate(idx, out=idx)
        left = left[idx]

        # Backward fill (propagation inverse)
        right = sector[::-1].copy()
        mask = np.isnan(right)
        idx = np.where(~mask, np.arange(len(right)), 0)
        np.maximum.accumulate(idx, out=idx)
        right = right[idx][::-1]

        # Combinaison (préférence au forward fill)
        sector_filled = np.where(np.isnan(left), right, left)

        # Conversion en millimètres
        return sector_filled * 1000.0

    # ========================================================================
    # OBSERVATION
    # ========================================================================
    
    def get_observation(self, init=False):
        """
        Construit l'observation pour l'agent.
        
        Args:
            init: True si c'est la première observation après reset
        """
        # Acquisition lidar avec retry
        tableau_lidar_mm = self.get_lidar_mm()
        retry_count = 0
        
        # Vérification que l'acquisition est valide (point central non nul)
        while tableau_lidar_mm[100] == 0 and retry_count < MAX_LIDAR_RETRY:
            self.nb_pb_acqui_lidar += 1
            print(f" Souci d'acquisition lidar #{self.nb_pb_acqui_lidar}")
            retry_count += 1
            tableau_lidar_mm = self.get_lidar_mm()

        # Normalisation : distances en [0, 1]
        current_lidar = (tableau_lidar_mm / LIDAR_RANGE_MM).astype(np.float32)
        current_lidar = np.clip(current_lidar, 0.0, 1.0)

        if init:
            # Première observation : pas d'historique
            previous_lidar = current_lidar.copy()
            current_speed = np.array([0.0], dtype=np.float32)
            previous_angle = np.array([0.0], dtype=np.float32)
        else:
            # Récupération de l'historique
            previous_lidar = self.observation["current_lidar"]
            current_speed = np.array(
                [self.consigne_vitesse / VITESSE_MAX_M_S], dtype=np.float32
            )
            previous_angle = np.array(
                [self.consigne_angle / MAXANGLE_DEGRE], dtype=np.float32
            )

        observation = {
            "current_lidar": current_lidar,
            "previous_lidar": previous_lidar,
            "current_speed": current_speed,
            "previous_angle": previous_angle,
        }

        self.observation = observation
        return observation

    # ========================================================================
    # CORRECTEUR P DE VITESSE
    # ========================================================================
    
    def P_correction_speed(self, min_distance_normalized):
        """
        Correcteur proportionnel : ajuste la vitesse selon la distance minimale.
        
        Args:
            min_distance_normalized: distance minimale normalisée [0, 1]
        
        Returns:
            vitesse en m/s
        """
        # Conversion en mètres
        min_distance_m = min_distance_normalized * LIDAR_RANGE_MM / 1000.0
        
        # Erreur par rapport à la distance cible
        erreur = min_distance_m - DISTANCE_CIBLE_M
        
        # Commande proportionnelle
        vitesse_commande = VITESSE_SECURITE_MIN + KP_VITESSE * erreur
        
        # Saturation
        vitesse_commande = np.clip(vitesse_commande, VITESSE_MIN_M_S, VITESSE_MAX_M_S)
        
        return vitesse_commande
    
    
    # ========================================================================
    # FONCTION DE RÉCOMPENSE
    # ========================================================================
    
    def get_reward(self, obs):
        """
        Calcule la récompense et détermine si l'épisode est terminé.
        
        Returns:
            reward (float), done (bool)
        """
        reward = 0.0
        done = False
        
        # Zone de détection frontale : [-40°, +40°] → indices [60, 140]
        front_sector = obs["current_lidar"][60:141]
        mini = np.min(front_sector)
        
        # ====================================================================
        # DÉTECTION DE COLLISION
        # ====================================================================
        
        
        if mini < SEUIL_COLLISION:
            self.numero_crash += 1
            reward = -100.0
            #done = True
            # print(f" COLLISION #{self.numero_crash} - Distance: {mini * LIDAR_RANGE_MM:.0f}mm")
        else:
            # ================================================================
            # RÉCOMPENSES POSITIVES
            # ================================================================
            
            # 1. Récompense de distance (logarithmique)
            # Log pour donner plus de poids aux petites distances
            mini_log = np.log(mini * 120 + 1e-6)  # +epsilon pour éviter log(0)
            
            # 2. Facteur de sécurité basé sur la distance
            safe_speed_factor = np.clip(mini_log / 1.94, 0.0, 1.0)  # 1.94 ≈ ln(7)
            
            # 3. Récompense de centrage (symétrie gauche/droite)
            # Indices 0-39 (gauche) vs 161-200 (droite symétrique)
            left_side = obs["current_lidar"][0:40]
            right_side = obs["current_lidar"][161:201][::-1]  # Inverser pour symétrie
            center_error = np.mean(np.abs(left_side - right_side))
            lane_reward = np.exp(-(center_error / 0.1) ** 2)
            
            # 4. Récompense d'orientation (symétrie zone centrale)
            # Indices 80-120 (gauche centrale) vs 81-121 (droite centrale)
            left_center = obs["current_lidar"][80:121]
            right_center = obs["current_lidar"][80:121][::-1]
            heading_error = np.mean(np.abs(left_center - right_center))
            heading_reward = np.exp(-(heading_error / 0.1) ** 2)
            
            # 5. Récompense de vitesse
            speed_normalized = obs["current_speed"][0]
            speed_ms = speed_normalized * VITESSE_MAX_M_S
            speed_reward = (3 * speed_ms - speed_ms ** 2) * safe_speed_factor
            
            # 6. Pénalité pour changements brusques d'angle
            angle_penalty = abs(self.consigne_angle - self.last_angle )/ (2*MAXANGLE_DEGRE)
            
            # ================================================================
            # RÉCOMPENSE TOTALE
            # ================================================================
            reward = (
                mini_log +
                3.0 * speed_reward +
                3.0 * lane_reward +
                1.5 * heading_reward -
                2   * angle_penalty
            )
            
            # Bonus de distance parcourue
            distance_parcourue = speed_ms * self.timestep_s
            self.total_distance += distance_parcourue
            reward += 0.1 * distance_parcourue  # Petit bonus pour encourager l'avancement

        # ====================================================================
        # CONDITION D'ARRÊT
        # ====================================================================
        self.reset_counter += 1
        if (self.reset_counter >= RESET_STEP) or (self.nb_pb_acqui_lidar > 100):
            done = True
            self.nb_pb_acqui_lidar = 0
            print(f" Épisode terminé - Steps: {self.reset_counter}, Distance: {self.total_distance:.1f}m")

        return reward, done

    # ========================================================================
    # STEP
    # ========================================================================
    
    def step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: array [angle_normalized] où angle ∈ [-1, 1]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # ====================================================================
        # 1. EXTRACTION DE L'ACTION (angle uniquement)
        # ====================================================================
        angle_normalized = float(action[0])
        self.consigne_angle = angle_normalized * MAXANGLE_DEGRE
        
        # Saturation (sécurité)
        self.consigne_angle = np.clip(self.consigne_angle, -MAXANGLE_DEGRE, MAXANGLE_DEGRE)
        
        # ====================================================================
        # 2. CALCUL DE LA VITESSE PAR CORRECTEUR P
        # ====================================================================
        # Obtenir la distance minimale depuis la dernière observation
        if self.observation is not None:
            front_sector = self.observation["current_lidar"][80:120]
            min_distance = np.min(front_sector)
        else:
            min_distance = 1.0  # Valeur par défaut
        
        self.consigne_vitesse = self.P_correction_speed(min_distance)
        
        # ====================================================================
        # 3. APPLICATION DES COMMANDES AU VÉHICULE
        # ====================================================================
        self.set_vitesse_m_s(self.consigne_vitesse)
        self.set_direction_degre(self.consigne_angle)
        
        # Avancer la simulation
        super().step()
        
        # ====================================================================
        # 4. NOUVELLE OBSERVATION ET RÉCOMPENSE
        # ====================================================================
        obs = self.get_observation()
        reward, done = self.get_reward(obs)
        self.last_angle=self.consigne_angle
        
        # Info supplémentaires pour debugging
        info = {
            "vitesse_ms": self.consigne_vitesse,
            "angle_degre": self.consigne_angle,
            "distance_min_m": min_distance * LIDAR_RANGE_MM / 1000.0,
            "total_distance": self.total_distance,
        }
        
        return obs, reward, done, False, info

    # ========================================================================
    # COMMANDES VÉHICULE
    # ========================================================================
    
    def set_vitesse_m_s(self, vitesse_m_s):
        """Définit la vitesse de croisière en m/s"""
        super().setCruisingSpeed(vitesse_m_s * 3.6)  # Conversion m/s → km/h

    def set_direction_degre(self, angle_degre):
        """Définit l'angle de braquage en degrés"""
        super().setSteeringAngle(-angle_degre * (math.pi / 180.0))  # Conversion → radians

    # ========================================================================
    # RESET
    # ========================================================================
    
    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        
        # Réinitialisation des commandes
        self.consigne_angle = 0.0
        self.last_angle = 0.0
        self.consigne_vitesse = 0.0
        self.set_vitesse_m_s(self.consigne_vitesse)
        self.set_direction_degre(self.consigne_angle)
        
        # Réinitialisation des métriques
        self.reset_counter = 0
        self.total_distance = 0.0
        
        # Avancer de quelques pas pour stabiliser la simulation
        for _ in range(20):
            super().step()
        
        # ====================================================================
        # GESTION DU CRASH (repositionnement du véhicule)
        # ====================================================================
        if self.numero_crash != 0:
            # Attendre l'arrêt complet
            while abs(super().getTargetCruisingSpeed()) > 0.001:
                super().step()
            
            # Demander repositionnement au superviseur
            self.packet_number += 1
            self.emitter.send(f"voiture crash {self.packet_number}")
            super().step()
            
            # Attendre confirmation du superviseur
            timeout = 0
            while self.receiver.getQueueLength() == 0 and timeout < 1000:
                self.set_vitesse_m_s(0.0)
                super().step()
                timeout += 1
            
            if self.receiver.getQueueLength() > 0:
                data = self.receiver.getString()
                self.receiver.nextPacket()
                print(f" Réponse superviseur: {data}")
            else:
                print("  Timeout superviseur")
            
            # S'assurer que le véhicule est arrêté
            self.consigne_vitesse = 0.0
            self.consigne_angle = 0.0
            self.last_angle = 0.0
            
            self.set_vitesse_m_s(0.0)
            self.set_direction_degre(0.0)
            
            while abs(super().getTargetCruisingSpeed()) >= 0.001:
                self.set_vitesse_m_s(self.consigne_vitesse)
                super().step()
        
        # Observation initiale
        observation = self.get_observation(init=True)
        
        return observation, {}


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    print("=" * 70)
    print("ENVIRONNEMENT WEBOTS RL - VERSION REFACTORISÉE")
    print("Action: Angle de braquage uniquement")
    print("Vitesse: Correcteur P basé sur distance minimale")
    print("=" * 70)
    
    t0 = time.time()
    
    # Création de l'environnement
    env = WebotsGymEnvironment()
    print(" Environnement créé")
    
    # Vérification de conformité Gymnasium
    check_env(env)
    print(" Vérification Gymnasium OK")
    """
    # ========================================================================
    # MODE 1 : ENTRAÎNEMENT
    # ========================================================================
    # Décommenter pour entraîner un nouveau modèle
    try:
        model = PPO.load("./model_A_with_cars5", env=env, device="cpu" ,learning_rate=3e-4)
        print(" Modèle chargé")
    except FileNotFoundError:
        print(" Modèle non trouvé.")
        return
    
    # model = PPO(
    #     policy="MultiInputPolicy",
    #     env=env,
    #     learning_rate=5e-4,
    #     verbose=1,
    #     device="cpu",
    #     tensorboard_log="./PPO_Tensorboard",
    # )
    print(" Début de l'apprentissage...")
    
    model.learn(total_timesteps=200000)
    
    t1 = time.time()
    print(f" Apprentissage terminé en {t1 - t0:.1f} secondes")
    model.save("./model_A_with_cars6")
    print(" Modèle sauvegardé ")
    """
    
    # ========================================================================
    # MODE 2 : TEST D'UN MODÈLE EXISTANT
    # ========================================================================
    try:
        model = PPO.load("./model_A_with_cars6", env=env, device="cpu")
        print(" Modèle chargé ")
    except FileNotFoundError:
        print(" Modèle non trouvé. Veuillez d'abord entraîner un modèle.")
        return
    
    print("\n Démarrage de la démonstration...")
    
    obs, _ = env.reset()
    cumul_reward = 0.0
    num_episodes = 0
    step_count = 0
    
    for i in range(50_000):
        # Prédiction de l'action
        action, _ = model.predict(obs, deterministic=True)
        
        # Exécution
        obs, reward, done, _, info = env.step(action)
        cumul_reward += reward
        step_count += 1
        
        # Affichage périodique
        if step_count % 500 == 0:
            print(f"Step {step_count:5d} | Reward cumulée: {cumul_reward:8.2f} | "
                  f"Vitesse: {info['vitesse_ms']:.2f} m/s | "
                  f"Angle: {info['angle_degre']:5.1f}° | "
                  f"Dist min: {info['distance_min_m']:.2f}m")
        
        if done:
            num_episodes += 1
            print(f"\n Épisode {num_episodes} terminé - Reward: {cumul_reward:.2f}")
            obs, _ = env.reset()
            cumul_reward = 0.0
    
    print("\n" + "=" * 70)
    print(f" Démonstration terminée - {num_episodes} épisodes complétés")
    print("=" * 70)
    #"""

if __name__ == "__main__":
    main()
