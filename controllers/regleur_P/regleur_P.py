from vehicle import Driver
from controller import Lidar
import numpy as np
import math

driver = Driver()

timestep = int(driver.getBasicTimeStep())

# ================================
# LIDAR (comme ton code RL)
# ================================
lidar = driver.getDevice("RpLidarA2")
lidar.enable(timestep)
lidar.enablePointCloud()

# ================================
# PARAMÈTRES
# ================================
KP = 0.8
DISTANCE_CIBLE = 1   # mètres
VITESSE_MIN = 0.3     # m/s

MAX_SPEED = 28         # km/h
MAX_ANGLE = 0.28       # rad

print("Correcteur P démarré")

# ================================
# FONCTION LIDAR (copiée RL)
# ================================
def get_lidar_m():
    raw = np.asarray(lidar.getRangeImage(), dtype=np.float64)

    # même secteur que ton RL
    sector = raw[80:281]

    # nettoyage
    sector = np.where((sector == 0) | np.isinf(sector), np.nan, sector)

    # remplacement simple des NaN
    if np.all(np.isnan(sector)):
        return np.ones(201) * 10.0

    return np.nan_to_num(sector, nan=10.0)


# ================================
# BOUCLE
# ================================
while driver.step() != -1:

    lidar_data = get_lidar_m()

    # ================================
    # DISTANCE DEVANT
    # ================================
    front = lidar_data[90:110]  # zone centrale

    distance = np.min(front)
    # print(distance)
    # ================================
    # CORRECTEUR P (vitesse)
    # ================================
    erreur = distance - DISTANCE_CIBLE
    vitesse = VITESSE_MIN + KP * erreur

    vitesse = np.clip(vitesse, 0.5, MAX_SPEED / 3.6)

    # ================================
    # DIRECTION (beaucoup plus stable)
    # ================================
    gauche = np.mean(lidar_data[140:180])
    droite = np.mean(lidar_data[20:60])

    erreur_direction = gauche - droite

    angle = 0.5 * erreur_direction
    angle = np.clip(angle, -MAX_ANGLE, MAX_ANGLE)

    # ================================
    # APPLICATION
    # ================================
    driver.setCruisingSpeed(vitesse * 3.6)
    driver.setSteeringAngle(angle)

