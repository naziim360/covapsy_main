"""
Webots Supervisor for CoVAPSy autonomous driving RL.

Main supervisor that:
- Manages vehicle repositioning after crashes
- Randomizes starting positions and directions
- Resets physics for new episodes
- Communicates with learning agent via messages
- Monitors vehicle safety bounds

This supervisor enables safe multi-episode RL training by handling
episode resets and vehicle repositioning.
"""

import random
import math
from controller import Supervisor  # type: ignore

from spawn import SpawnManager, position_vehicle
from checkpoints import CheckpointTracker, DEFAULT_CHECKPOINTS_TRAIN
from utils import is_position_valid, get_safe_position, create_rotation_array


# Configuration
USE_TEST_TRACK: bool = False
NB_SPARRING_PARTNER_CARS: int = 0
RECEIVER_SAMPLING_PERIOD_MS: int = 64
SAFETY_CHECK_BOUNDS: dict = {
    "max_x": 20.0,
    "max_y": 20.0,
    "max_z": 0.1,
}


def main():
    """Main supervisor loop."""
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    print("[Supervisor] Starting CoVAPSy Supervisor")
    
    supervisor = Supervisor()
    basic_timestep = int(supervisor.getBasicTimeStep())
    
    # ========================================================================
    # SETUP COMMUNICATION
    # ========================================================================
    
    receiver = supervisor.getDevice("receiver")
    receiver.enable(RECEIVER_SAMPLING_PERIOD_MS)
    
    emitter = supervisor.getDevice("emitter")
    packet_number = 0
    
    # ========================================================================
    # GET VEHICLE NODES
    # ========================================================================
    
    # Main learning car
    TT02_DEF = "TT02_2023b_RL"
    tt_02 = supervisor.getFromDef(TT02_DEF)
    if tt_02 is None:
        raise RuntimeError(
            f"[Supervisor] DEF '{TT02_DEF}' not found in world file (.wbt)"
        )
    
    tt_02_translation = tt_02.getField("translation")
    tt_02_rotation = tt_02.getField("rotation")
    
    # Sparring partner cars
    sparringpartner_cars = []
    sparringpartner_translations = []
    sparringpartner_rotations = []
    
    for i in range(NB_SPARRING_PARTNER_CARS):
        def_name = f"sparringpartner_car_{i}"
        node = supervisor.getFromDef(def_name)
        if node is None:
            raise RuntimeError(
                f"[Supervisor] DEF '{def_name}' not found in world file"
            )
        sparringpartner_cars.append(node)
        sparringpartner_translations.append(node.getField("translation"))
        sparringpartner_rotations.append(node.getField("rotation"))
    
    # ========================================================================
    # INITIALIZE MANAGERS
    # ========================================================================
    
    spawn_manager = SpawnManager(use_test_track=USE_TEST_TRACK)
    checkpoint_tracker = CheckpointTracker(DEFAULT_CHECKPOINTS_TRAIN)
    
    error_count = 0
    
    print("[Supervisor] Initialization complete")
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    print("[Supervisor] Main loop started\n")
    
    while supervisor.step(basic_timestep) != -1:
        
        # ====================================================================
        # SAFETY CHECK: Monitor vehicle position
        # ====================================================================
        
        current_position = tt_02_translation.getSFVec3f()
        x, y, z = current_position[0], current_position[1], current_position[2]
        
        # Check if position is valid
        if not is_position_valid(x, y, z, **SAFETY_CHECK_BOUNDS):
            error_count += 1
            print(
                f"[Supervisor] Invalid position detected #{error_count}: "
                f"({x:.2f}, {y:.2f}, {z:.2f})"
            )
            
            # Reposition to safe location
            safe_x, safe_y, safe_z, safe_angle = get_safe_position()
            tt_02_translation.setSFVec3f([safe_x, safe_y, safe_z])
            tt_02_rotation.setSFRotation(create_rotation_array(safe_angle))
            supervisor.simulationResetPhysics()
            
            # Step to apply changes
            supervisor.step(basic_timestep)
        
        # ====================================================================
        # MESSAGE HANDLING: Process reset requests from agent
        # ====================================================================
        
        if receiver.getQueueLength() > 0:
            try:
                message = receiver.getString()
                receiver.nextPacket()
                print(f"[Supervisor] Received message: {message}")
                
                # Process message (crash reset request)
                if "crash" in message.lower():
                    _reset_all_vehicles(
                        supervisor,
                        spawn_manager,
                        tt_02_translation,
                        tt_02_rotation,
                        sparringpartner_translations,
                        sparringpartner_rotations,
                    )
                    
                    # Send acknowledgment
                    packet_number += 1
                    ack_msg = f"voiture replacee num : {packet_number}"
                    emitter.send(ack_msg.encode("utf-8"))
                    
            except Exception as e:
                print(f"[Supervisor] Error processing message: {e}")


def _reset_all_vehicles(supervisor: Supervisor,
                       spawn_manager: SpawnManager,
                       learning_car_translation,
                       learning_car_rotation,
                       sparring_translations: list,
                       sparring_rotations: list) -> None:
    """
    Reset all vehicles to new random positions.
    
    Args:
        supervisor: Webots Supervisor instance
        spawn_manager: SpawnManager for position selection
        learning_car_translation: Main car translation field
        learning_car_rotation: Main car rotation field
        sparring_translations: List of sparring car translation fields
        sparring_rotations: List of sparring car rotation fields
    """
    # ========================================================================
    # SELECT RANDOM POSITIONS AND DIRECTION
    # ========================================================================
    
    direction = spawn_manager.randomize_direction()
    num_cars = 1 + NB_SPARRING_PARTNER_CARS
    position_indices = spawn_manager.select_random_positions(num_cars)
    
    print(f"[Supervisor] Resetting vehicles - Direction: {'reversed' if direction else 'normal'}")
    
    # ========================================================================
    # POSITION LEARNING CAR
    # ========================================================================
    
    learning_pos_idx = position_indices[0]
    spawn_x, spawn_y, spawn_z, spawn_angle = spawn_manager.get_spawn_point(
        learning_pos_idx,
        direction=direction,
    )
    
    learning_car_translation.setSFVec3f([spawn_x, spawn_y, spawn_z])
    learning_car_rotation.setSFRotation(create_rotation_array(spawn_angle))
    
    print(
        f"  Learning car: pos_idx={learning_pos_idx}, "
        f"({spawn_x:.2f}, {spawn_y:.2f}), angle={spawn_angle:.2f}rad"
    )
    
    # ========================================================================
    # POSITION SPARRING PARTNER CARS
    # ========================================================================
    
    for i in range(NB_SPARRING_PARTNER_CARS):
        sparring_pos_idx = position_indices[i + 1]
        spawn_x, spawn_y, spawn_z, spawn_angle = spawn_manager.get_spawn_point(
            sparring_pos_idx,
            direction=direction,
        )
        
        sparring_translations[i].setSFVec3f([spawn_x, spawn_y, spawn_z])
        sparring_rotations[i].setSFRotation(create_rotation_array(spawn_angle))
        
        print(
            f"  Sparring car {i}: pos_idx={sparring_pos_idx}, "
            f"({spawn_x:.2f}, {spawn_y:.2f}), angle={spawn_angle:.2f}rad"
        )
    
    # ========================================================================
    # RESET PHYSICS AND STABILIZE
    # ========================================================================
    
    supervisor.simulationResetPhysics()
    
    # Stabilization steps
    basic_timestep = int(supervisor.getBasicTimeStep())
    for _ in range(20):
        supervisor.step(basic_timestep)
    
    print("[Supervisor] Vehicle reset complete\n")


if __name__ == "__main__":
    main()
