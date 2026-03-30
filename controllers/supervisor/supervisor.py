"""
Supervisor CoVAPSy (Webots) - Race Line Formation Reset
- Repositions RL car + sparring partners in a racing line formation
- Randomizes car order (RL car can be first, middle, or last)
- All cars face the same direction on the same track section
- Maintains safe spacing between cars
- Measures lap times using a line start/finish and sends them to the RL controller
- Tracks lap count per episode and notifies agent after each lap
"""

import random
import math
from controller import Supervisor  # type: ignore


# -----------------------------
# Global constants
# -----------------------------
PI = math.pi
RECEIVER_SAMPLING_PERIOD = 64  # ms
NB_SPARRING_PARTNER_CARS = 1

CAR_SPACING = 1.5            # metres between cars (bumper to bumper)
LATERAL_OFFSET_RANGE = 0.15  # ±15 cm lateral jitter for realism

USE_TEST_TRACK = False

# -----------------------------
# Lap timer — finish line
# -----------------------------
# Place this segment ON the track the cars actually drive on.
# TRAIN: cars pass through ~(0, -0.5) heading right → vertical line at x=0
# TEST:  cars pass through ~(2.0, 5.5) heading right → vertical line at x=2
if USE_TEST_TRACK:
    LINE_START = (2.0, 5.0)
    LINE_END   = (2.0, 6.0)
else:
    LINE_START = (0.0, -1.0)
    LINE_END   = (0.0,  0.0)


# -----------------------------
# Helper functions
# -----------------------------
def angle_clip(a: float) -> float:
    """Normalise angle to [-π, π] (Webots convention)."""
    a = a % (2 * PI)
    return a if a <= PI else a - 2 * PI

def cross_2d(ax, ay, bx, by):
    return ax * by - ay * bx

def segments_intersect(p1, p2, p3, p4):
    """Return (True, point) if segment p1→p2 crosses p3→p4, else (False, None)."""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (p4[0] - p3[0], p4[1] - p3[1])
    denom = cross_2d(d1[0], d1[1], d2[0], d2[1])
    if abs(denom) < 1e-10:
        return False, None
    diff = (p3[0] - p1[0], p3[1] - p1[1])
    t = cross_2d(diff[0], diff[1], d2[0], d2[1]) / denom
    u = cross_2d(diff[0], diff[1], d1[0], d1[1]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, (p1[0] + t * d1[0], p1[1] + t * d1[1])
    return False, None

def get_forward(orientation):
    """Extract world-space forward (dx, dy) from Webots 3×3 orientation matrix."""
    return (orientation[0], orientation[3])

def reset_lap_timer():
    global lap_start_time, last_pos, lap_count
    lap_start_time = None
    last_pos       = None
    lap_count      = 0


# -----------------------------
# Race line definitions
# -----------------------------
race_lines_train = [
    {"center": [ 0.0, -0.5], "angle": 0.00,  "direction": [1.0, 0.0], "name": "Center Straight"},
    {"center": [-1.0, -2.5], "angle": 0.00,  "direction": [1.0, 0.0], "name": "Bottom Straight"},
    {"center": [-4.2,  0.0], "angle": PI/2,  "direction": [0.0, 1.0], "name": "Left Vertical"},
]

race_lines_test = [
    {"center": [ 2.0,  5.5], "angle":  0.00, "direction": [ 1.0,  0.0], "name": "Main Straight"},
    {"center": [ 5.5,  0.0], "angle": -PI/2, "direction": [ 0.0, -1.0], "name": "Right Vertical"},
    {"center": [ 1.5, -4.7], "angle":  PI/2, "direction": [ 0.0,  1.0], "name": "Bottom Vertical"},
    {"center": [-0.5,  0.6], "angle": -PI/2, "direction": [ 0.0, -1.0], "name": "Left Vertical"},
]

race_lines = race_lines_test if USE_TEST_TRACK else race_lines_train


# -----------------------------
# Supervisor init
# -----------------------------
supervisor    = Supervisor()
basicTimeStep = int(supervisor.getBasicTimeStep())

receiver = supervisor.getDevice("receiver")
receiver.enable(RECEIVER_SAMPLING_PERIOD)
emitter = supervisor.getDevice("emitter")
packet_number = 0

TT02_DEF = "TT02_2023b_RL"
tt_02 = supervisor.getFromDef(TT02_DEF)
if tt_02 is None:
    raise RuntimeError(f"Supervisor: DEF '{TT02_DEF}' not found in world (.wbt).")

tt_02_translation = tt_02.getField("translation")
tt_02_rotation    = tt_02.getField("rotation")

sparringpartner_car_translation_fields = []
sparringpartner_car_rotation_fields    = []

for i in range(NB_SPARRING_PARTNER_CARS):
    def_name = f"sparringpartner_car_{i}"
    node = supervisor.getFromDef(def_name)
    if node is None:
        raise RuntimeError(f"Supervisor: DEF '{def_name}' not found in world (.wbt).")
    sparringpartner_car_translation_fields.append(node.getField("translation"))
    sparringpartner_car_rotation_fields.append(node.getField("rotation"))

erreur_position = 0
reset_count     = 0

# Lap timer state
lap_start_time = None
last_pos       = None
lap_count      = 0   # laps completed in the current episode


# -----------------------------
# Reset procedure
# -----------------------------
def reset_all_cars_in_line():
    global packet_number, reset_count

    reset_lap_timer()   # also resets lap_count to 0
    reset_count += 1
    total_cars = 1 + NB_SPARRING_PARTNER_CARS

    race_line   = random.choice(race_lines)
    reverse     = random.choice([False, True])
    rl_position = random.randint(0, total_cars - 1)

    center_x, center_y = race_line["center"]
    base_angle          = race_line["angle"]
    dir_x, dir_y        = race_line["direction"]

    if reverse:
        dir_x, dir_y = -dir_x, -dir_y
        base_angle   = angle_clip(base_angle + PI)

    perp_x, perp_y  = -dir_y, dir_x
    total_length     = (total_cars - 1) * CAR_SPACING
    first_car_offset = -total_length / 2.0

    car_positions = []
    for i in range(total_cars):
        along   = first_car_offset + i * CAR_SPACING
        lateral = random.uniform(-LATERAL_OFFSET_RANGE, LATERAL_OFFSET_RANGE)
        x = center_x + dir_x * along + perp_x * lateral
        y = center_y + dir_y * along + perp_y * lateral
        angle = angle_clip(base_angle + random.uniform(-math.pi / 36, math.pi / 36))
        car_positions.append({"pos": [x, y, 0.04], "angle": angle, "is_rl": (i == rl_position)})

    print(f"[Supervisor] Reset #{reset_count} | Section: {race_line['name']} | "
          f"RL position: {rl_position + 1}/{total_cars} | "
          f"Direction: {'REVERSE' if reverse else 'FORWARD'}")

    rl_data = car_positions[rl_position]
    tt_02_translation.setSFVec3f(rl_data["pos"])
    tt_02_rotation.setSFRotation([0, 0, 1, rl_data["angle"]])

    sp_index = 0
    for car_data in car_positions:
        if not car_data["is_rl"] and sp_index < NB_SPARRING_PARTNER_CARS:
            sparringpartner_car_translation_fields[sp_index].setSFVec3f(car_data["pos"])
            sparringpartner_car_rotation_fields[sp_index].setSFRotation(
                [0, 0, 1, car_data["angle"]])
            sp_index += 1

    supervisor.simulationResetPhysics()
    for _ in range(20):
        supervisor.step(basicTimeStep)

    packet_number += 1
    emitter.send(f"voiture replacee num : {packet_number}".encode("utf-8"))


# -----------------------------
# Main loop
# -----------------------------
print("[Supervisor] RACE LINE MODE - STARTED")
print(f"[Supervisor] Total cars      : {1 + NB_SPARRING_PARTNER_CARS}")
print(f"[Supervisor] Car spacing     : {CAR_SPACING} m")
print(f"[Supervisor] Track           : {'TEST' if USE_TEST_TRACK else 'TRAIN'}")
print(f"[Supervisor] Race lines      : {len(race_lines)}")
print(f"[Supervisor] Finish line     : {LINE_START} → {LINE_END}")

while supervisor.step(basicTimeStep) != -1:

    # ── Safety check ───────────────────────────────────────────────────
    pos1 = tt_02_translation.getSFVec3f()
    if abs(pos1[0]) > 20 or abs(pos1[1]) > 20 or abs(pos1[2]) > 0.1:
        tt_02_rotation.setSFRotation([0, 0, 1, -PI / 2])
        tt_02_translation.setSFVec3f([2.98, 2.0, 0.04])
        supervisor.simulationResetPhysics()
        erreur_position += 1
        print(f"[Supervisor] Aberrant position #{erreur_position}, safety reset.")
        supervisor.step(basicTimeStep)
        reset_lap_timer()
        continue

    # ── Lap timer ──────────────────────────────────────────────────────
    pos         = tt_02_translation.getSFVec3f()
    pos_2d      = (pos[0], pos[1])
    orientation = tt_02.getOrientation()
    heading     = get_forward(orientation)

    if last_pos is not None:
        intersect, _ = segments_intersect(last_pos, pos_2d, LINE_START, LINE_END)

        if intersect:
            move_dir = (pos_2d[0] - last_pos[0], pos_2d[1] - last_pos[1])
            moving_forward = (move_dir[0] * heading[0] + move_dir[1] * heading[1]) > 0

            if moving_forward:
                now = supervisor.getTime()

                if lap_start_time is not None:
                    # ── Completed lap ───────────────────────────────────
                    lap_time = now - lap_start_time
                    lap_count += 1

                    print(f"[Supervisor] Lap {lap_count} time: {lap_time:.2f}s")

                    # Notify agent: lap number + time
                    emitter.send(f"LAP {lap_count} {lap_time:.4f}".encode("utf-8"))

                lap_start_time = now

    last_pos = pos_2d

    # ── Reset requests from RL agent ───────────────────────────────────
    while receiver.getQueueLength() > 0:
        try:
            data = receiver.getString()
            receiver.nextPacket()
            print(f"[Supervisor] Message received: {data!r}")
        except Exception as e:
            print(f"[Supervisor] Reception error: {e}")
        reset_all_cars_in_line()