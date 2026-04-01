"""
Supervisor CoVAPSy (Webots) - Reset multi-voitures pour RL
- Repositionne la voiture apprenante + N sparring partners après réception d'un message (collision)
- Randomise sens de circulation + points de départ
- Ajoute une stabilisation (20 steps)
"""

import random
import math
from controller import Supervisor # type: ignore



# -----------------------------
# Global constants
# -----------------------------
PI = math.pi
RECEIVER_SAMPLING_PERIOD = 64  # ms
NB_SPARRING_PARTNER_CARS = 0

# Choisir la piste (train/test)
USE_TEST_TRACK = False

# -----------------------------
# Helper functions
# -----------------------------
def value_clip(x: float, low: float, up: float) -> float:
    """Clamp x to [low, up]."""
    return low if x < low else up if x > up else x

def angle_clip(a: float) -> float:
    """
    Webots rotation angle should be in [-pi, pi].
    """
    a = a % (2 * PI)
    return a if a <= PI else a - 2 * PI

# -----------------------------
# Starting positions
# Each entry: [[x_min, x_max], [y_min, y_max], base_angle]
# NOTE: Here we keep z constant (like your code), and randomize x,y.
# -----------------------------
starting_positions_train = [
    [ [ -1.5,  1.5],[ -0.225,  -0.78],  0.00], #centre
    [ [ -3.74, 1.74],[ -1.84,-3.13], 0.00], #en bas
    [ [-4.6,-3.7],[ -2.5,  2.5], PI/2], #gauche
    [ [3.2,4.5],[ 3.18,  3.3],  PI/2] #haut droite
]
# starting_positions_train = [
#     [[ 0.00,  4.00], [ 5.10,  5.30],  0.00],
#     [[ 4.95,  5.15], [ 0.30,  4.10], -PI/2],
#     [[ 2.70,  4.50], [-1.25, -0.75],  PI],
#     [[ 3.20,  3.45], [ 2.00,  2.40],  PI/2],
#     [[ 1.90,  2.70], [ 3.10,  3.30],  PI],
#     [[ 0.00,  0.25], [-0.50,  1.50], -PI/2],
#     [[-2.20, -1.90], [-0.50,  2.30],  PI/2],
# ]

starting_positions_test = [
    [[ 0.60,  4.20], [ 5.40,  5.60],  0.00],
    [[ 5.40,  5.60], [-4.30,  4.30], -PI/2],
    [[ 1.40,  1.60], [-5.00, -4.30],  PI/2],
    [[ 3.40,  3.60], [-2.85, -1.45],  PI/2],
    [[ 3.40,  3.60], [ 2.20,  2.50],  PI/2],
    [[-0.60, -0.40], [ 0.30,  0.90], -PI/2],
    [[-3.20, -2.00], [-4.60, -4.40],  PI],
    [[-2.60, -2.40], [-0.80,  2.30],  PI/2],
]

starting_positions = starting_positions_test if USE_TEST_TRACK else starting_positions_train

# -----------------------------
# Init supervisor
# -----------------------------
supervisor = Supervisor()
basicTimeStep = int(supervisor.getBasicTimeStep())

# Receiver / Emitter (must exist as devices on the Supervisor robot)
receiver = supervisor.getDevice("receiver")
receiver.enable(RECEIVER_SAMPLING_PERIOD)

emitter = supervisor.getDevice("emitter")
packet_number = 0

# -----------------------------
# Get nodes by DEF
# -----------------------------
TT02_DEF = "TT02_2023b_RL"
tt_02 = supervisor.getFromDef(TT02_DEF)
if tt_02 is None:
    raise RuntimeError(f"Supervisor: DEF '{TT02_DEF}' introuvable dans le monde (.wbt).")

tt_02_translation = tt_02.getField("translation")
tt_02_rotation = tt_02.getField("rotation")

sparringpartner_car_nodes = []
sparringpartner_car_translation_fields = []
sparringpartner_car_rotation_fields = []

for i in range(NB_SPARRING_PARTNER_CARS):
    def_name = f"sparringpartner_car_{i}"
    node = supervisor.getFromDef(def_name)
    if node is None:
        raise RuntimeError(f"Supervisor: DEF '{def_name}' introuvable dans le monde (.wbt).")
    sparringpartner_car_nodes.append(node)
    sparringpartner_car_translation_fields.append(node.getField("translation"))
    sparringpartner_car_rotation_fields.append(node.getField("rotation"))

# Compteur de positions aberrantes
erreur_position = 0

# -----------------------------
# Reset procedure
# -----------------------------
def reset_all_cars():
    global packet_number

    # Choix du sens de circulation
    direction = random.choice([0, 1])  # 0: sens de base, 1: inverse

    # Choisir 1 + N positions distinctes
    indices = random.sample(range(len(starting_positions)), 1 + NB_SPARRING_PARTNER_CARS)

    # --- Place learning car ---
    coords = starting_positions[indices[0]]
    #print(f"[Supervisor] Chosen starting position index for learning car: {indices[0]}")
    start_x = random.uniform(coords[0][0], coords[0][1])
    start_y = random.uniform(coords[1][0], coords[1][1])
    start_z = 0.04  # constant, like your code

    base_angle = coords[2]
    start_angle = random.uniform(base_angle - PI/12, base_angle + PI/12)

    if direction == 1:
        start_angle += PI  # reverse direction
    #print(f"[Supervisor] Reset learning car to x={start_x:.2f}, y={start_y:.2f}, angle={start_angle:.2f} rad")
    tt_02_rotation.setSFRotation([0, 0, 1, angle_clip(start_angle)])
    tt_02_translation.setSFVec3f([start_x, start_y, start_z])

    #--- Place sparring partner cars ---
    for i in range(NB_SPARRING_PARTNER_CARS):
        coords = starting_positions[indices[i + 1]]

        sx = random.uniform(coords[0][0], coords[0][1])
        sy = random.uniform(coords[1][0], coords[1][1])
        sz = 0.04

        base_angle = coords[2]
        ang = random.uniform(base_angle - PI/12, base_angle + PI/12)
        if direction == 1:
            ang += PI

        sparringpartner_car_translation_fields[i].setSFVec3f([sx, sy, sz])
        sparringpartner_car_rotation_fields[i].setSFRotation([0, 0, 1, angle_clip(ang)])
    
    supervisor.simulationResetPhysics()
    # Attente stabilisation
    for _ in range(20):
        supervisor.step(basicTimeStep)

    # Ack vers l'agent
    packet_number += 1
    

    msg = f"voiture replacee num : {packet_number}"
    emitter.send(msg.encode("utf-8"))

# -----------------------------
# Main loop
# -----------------------------
print("[Supervisor] STARTED")

#last_periodic_reset_sec = -1  # mémorise la dernière seconde où on a reset
# 1) Reset périodique toutes les 5s (1 seule fois)
#    now_sec = int(supervisor.getTime())
#    if now_sec % 5 == 0 and now_sec != last_periodic_reset_sec:
#        last_periodic_reset_sec = now_sec
#        print(f"[Supervisor] periodic reset at t={now_sec}s")
#       reset_all_cars()

while supervisor.step(basicTimeStep) != -1:

    
    # Détection positions incohérentes (sécurité)
    pos1 = tt_02_translation.getSFVec3f()

    # pos = [x, y, z]
    if abs(pos1[0]) > 20 or abs(pos1[1]) > 20 or abs(pos1       [2]) > 0.1:
        # Replace à un point sûr (comme ton code)
        safe_rot = [0, 0, 1, -PI/2]
        safe_pos = [2.98, 2, 0.04]
        tt_02_rotation.setSFRotation(safe_rot)
        tt_02_translation.setSFVec3f(safe_pos)
        supervisor.simulationResetPhysics()

        erreur_position += 1
        print(f"[Supervisor] Position aberrante détectée #{erreur_position}, repositionnement au point sûr.")
        # petit step pour appliquer
        supervisor.step(basicTimeStep)

    # Reset signal from agent
    if receiver.getQueueLength() > 0:
        try:
            data = receiver.getString()
            receiver.nextPacket()
            print(f"[Supervisor] message recu: {data}")
            
        except Exception:
            print("[Supervisor] souci de reception")
        reset_all_cars()
        
