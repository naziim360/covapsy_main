# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2023b
# Inspiré de vehicle_driver_altino controller
# Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Taynaud
# juillet 2023

from vehicle import Driver # type: ignore
from controller import Lidar # type: ignore
import math
import matplotlib.pyplot as plt
import numpy as np
import math

    
def discrete_sectors(distances):
    distances = np.asarray(distances)

    # Extract angles [-100 .. +100] → 201 values
    sector = distances[80:281]

    # Clamp distances (invalid → max range)
    sector_mm = np.where((sector > 0) & (sector < 20.0),
                          sector,
                          20.0)

    # ----- HARD-CODED 5 SECTORS -----
    step = 201 // 5  # 40

    d0 = sector_mm[0*step : 1*step].min()
    d1 = sector_mm[1*step : 2*step].min()
    d2 = sector_mm[2*step : 3*step].min()
    d3 = sector_mm[3*step : 4*step].min()
    d4 = sector_mm[4*step : 201   ].min()

    discrete = np.array([d0, d1, d2, d3, d4], dtype=np.float64)

    # Sector center angles (degrees)
    angles_rad = [
        math.radians(-100 + step * 0.5),
        math.radians(-100 + step * 1.5),
        math.radians(-100 + step * 2.5),
        math.radians(-100 + step * 3.5),
        math.radians(-100 + step * 4.5),
    ]

    # ----- Plot discrete points -----
    xs, ys = [], []
    for r, a in zip(discrete, angles_rad):
        if r <= 0.0 or math.isinf(r):
            continue
        xs.append(-r * math.cos(a))
        ys.append(r * math.sin(a))

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=50)
    plt.plot(0, 0, "ro")  # LiDAR position
    plt.axis("equal")
    plt.grid(True)
    plt.title("LiDAR – 5 discrete sectors")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()
    

def plot_lidar(distances, max_range=None):
    """
    distances : liste de distances (1 point = 1 degré, sens horaire)
    max_range : distance max d'affichage (optionnel)
    """
    xs = []
    ys = []
    
    for angle_deg, r in enumerate(distances[80:281]):  # angles de -100 à +100 degrés
        if r <= 0.0 or math.isinf(r):
            continue
        if max_range is not None and r > max_range:
            continue
        
        angle_rad = math.radians(angle_deg) + math.pi  # le lidar est orienté vers l'arrière de la voiture, on ajoute donc 180° pour que 0° soit devant la voiture

        x = -r * math.cos(angle_rad)
        y = r * math.sin(angle_rad)

        xs.append(x)
        ys.append(y)
    
        
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=5)
    plt.plot(0, 0, "ro")  # position du LiDAR

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("LiDAR scan (vue du dessus)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()


driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

#Lidar
lidar = Lidar("RpLidarA2")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud() 

#clavier
keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

# vitesse en km/h
speed = 0
maxSpeed = 28 #km/h

# angle de la direction
angle = 0
maxangle = 0.28 #rad (étrange, la voiture est défini pour une limite à 0.31 rad...

# mise a zéro de la vitesse et de la direction
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
# mode manuel et mode auto desactive
modeManuel = False
modeAuto = False
print("cliquer sur la vue 3D pour commencer")
print("m pour mode manuel, a pour mode auto, n pour stop, l pour affichage données lidar")
print("en mode manuel utiliser les flèches pour accélérer, freiner et diriger")

while driver.step() != -1:

    speed = driver.getTargetCruisingSpeed()

    while True:
        #acquisition des donnees du lidar
        donnees_lidar = lidar.getRangeImage()
                
        # recuperation de la touche clavier
        currentKey = keyboard.getKey()
        if currentKey == -1:
            break
        if currentKey == ord('m') or currentKey == ord('M'):
            if not modeManuel:
                modeManuel = True
                modeAuto = False
                print("------------Mode Manuel Activé---------------")
        elif currentKey == ord('n') or currentKey == ord('N'):
            if modeManuel or modeAuto :
                modeManuel = False
                modeAuto = False
                print("--------Modes Manuel et Auto Désactivé-------")
        elif currentKey == ord('a') or currentKey == ord('A'):
            if not modeAuto : 
                modeAuto = True
                modeManuel = False
                print("------------Mode Auto Activé-----------------")
        elif currentKey == ord('l') or currentKey == ord('L'):
                #print("-----donnees du lidar en metres sens horaire au pas de 1°-----")
                # for i in range(len(donnees_lidar)) :
                #     print(f"{donnees_lidar[i]:.3f}   ", end='')
                #     if (i+1)%10 == 0 :        
                #        print()
                # print("------------------------------------------------------------")
                # print("Nb points lidar:", len(donnees_lidar))
                # for i in range(len(donnees_lidar)):
                #     if donnees_lidar[i] == min(donnees_lidar):
                #         print(f"angle {i}° : distance minimale {donnees_lidar[i]:.3f} m")
                # print(f"angle {0}° : distance {donnees_lidar[0]:.3f} m")
                # print(f"angle {90}° : distance {donnees_lidar[90]:.3f} m")
                # print(f"angle {180}° : distance {donnees_lidar[180]:.3f} m")
                # print(f"angle {270}° : distance {donnees_lidar[270]:.3f} m")
                plot_lidar(donnees_lidar)
                discrete_sectors(donnees_lidar)
      
        # Controle en mode manuel
        if modeManuel:
            if currentKey == keyboard.UP:
                speed += 0.2
            elif currentKey == keyboard.DOWN:
                speed -= 0.2
            elif currentKey == keyboard.LEFT:
                angle -= 0.04
            elif currentKey == keyboard.RIGHT:
                angle += 0.04

    if not modeManuel and not modeAuto:
        speed = 0
        angle = 0
        
    if modeAuto:
        speed = 3 #km/h
        #l'angle de la direction est la différence entre les mesures des rayons 
        #du lidar à (-99+18*2)=-63° et (-99+81*2)=63°
        angle = donnees_lidar[240]-donnees_lidar[120]

    # clamp speed and angle to max values
    if speed > maxSpeed:
        speed = maxSpeed
    elif speed < -1 * maxSpeed:
        speed = -1 * maxSpeed
    if angle > maxangle:
        angle = maxangle
    elif angle < -maxangle:
        angle = -maxangle

    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(angle)

# def get_lidar_mm(): (not important)
#         raw = np.array(lidar.getRangeImage(), dtype=np.float64)  # meters, shape (360,)
#         # Mirror the raw data
#         #raw_mirrored = raw[::-1]
#         # for i in range(1, 360):
#         #     print(f"raw[{i}] = " + str(raw[i])) 
#         #     print(f"raw_mirrored[{360-i}] = " + str(raw_mirrored[360-i])) 

#         # Extract angles [-100 .. +100] => 201 values
#         sector = raw[80:281]
        

#         # Convert to mm + filter
#         sector_mm = np.where((sector > 0) & (sector < 20),
#                             sector * 1000.0,
#                             0.0)

#         # for i in range(0, 201):
#         #     print(f"sector[{i}] = " + str(sector[i])) 
#         #     print(f"sector_mm[{i}] = " + str(sector_mm[i]))
#         plt.plot(raw)
#         plt.show() 
#         return np.asarray(sector_mm, dtype=np.float64)
                          

# def plot_lidar(sector_mm, max_range=None):
#     xs, ys = [], []

#     angle_deg = -100
#     for r in sector_mm:
#         if r <= 0 or math.isinf(r):
#             angle_deg += 1
#             continue
#         if max_range is not None and r > max_range:
#             angle_deg += 1
#             continue

#         angle_rad = math.radians(angle_deg)

#         x = r * math.cos(angle_rad)
#         y = r * math.sin(angle_rad)

#         xs.append(x)
#         ys.append(y)

#         angle_deg += 1

#     # plt.figure(figsize=(6, 6))
#     # plt.scatter(xs, ys, s=5)
#     # plt.axis("equal")
#     # plt.xlabel("x (mm)")
#     # plt.ylabel("y (mm)")
#     # plt.title("Lidar view")
#     # plt.grid()
#     # plt.show()