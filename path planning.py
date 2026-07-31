import glob
import os
import sys
import random
import time
import math

# 1. CARLA PATH SETUP
try:
    carla_egg_path = glob.glob('D:/carla/carla simulation/WindowsNoEditor/PythonAPI/carla/dist/carla-*3.7-win-amd64.egg')[0]
    sys.path.append(carla_egg_path)
except IndexError:
    print("Error: .egg file path check!")
    sys.exit()

import carla

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0) 

    try:
        print("Highway Map Loading...")
        world = client.load_world('Town04') 
        
        # --- SHAKING FIX: Synchronous Mode & Fixed Delta ---
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS fixed for physics stability
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 2. SPAWN MAIN VEHICLE
        bp = blueprint_library.filter('model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        start_spawn = random.choice(spawn_points)
        vehicle = world.spawn_actor(bp, start_spawn)
        
        # 3. TRAFFIC MANAGER (Speed & Overtake Logic)
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        tm.set_global_distance_to_leading_vehicle(3.0)
        
        # --- SPEED LIMIT FIX (50 MPH = 80 km/h) ---
        # Highway limit 100-la irundhu 20% koraicha ~80 km/h (50 MPH) varum
        tm.vehicle_percentage_speed_difference(vehicle, 20.0) 
        tm.auto_lane_change(vehicle, True) 
        vehicle.set_autopilot(True, tm.get_port())

        # Heavy Traffic (200 Cars)
        print("Spawning Heavy Traffic...")
        for i in range(200):
            t_bp = random.choice(blueprint_library.filter('vehicle.*'))
            t_spawn = random.choice(spawn_points)
            t_veh = world.try_spawn_actor(t_bp, t_spawn)
            if t_veh:
                # Traffic cars medhuva poga veppom (30-40 MPH)
                tm.vehicle_percentage_speed_difference(t_veh, 40.0)
                t_veh.set_autopilot(True, tm.get_port())

        # 4. CAMERA SETUP (Rigid Attachment to stop shaking)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_trans = carla.Transform(carla.Location(x=-8.0, z=3.5), carla.Rotation(pitch=-15))
        camera = world.spawn_actor(camera_bp, camera_trans, attach_to=vehicle, 
                                   attachment_type=carla.AttachmentType.Rigid)
        
        spectator = world.get_spectator()

        # 5. LOOP TRACKING & UI
        loop_target = 6946.0 
        total_dist_covered = 1.0 
        last_loc = vehicle.get_location()

        print("Simulation Running... No Shaking Mode.")

        while total_dist_covered < loop_target:
            world.tick() # Physics update
            
            # --- SHAKING REMOVAL ---
            # Spectator camera transform-ai tick-ku apparam update pannanum
            spectator.set_transform(camera.get_transform())
            
            curr_loc = vehicle.get_location()
            step_dist = math.sqrt((curr_loc.x - last_loc.x)**2 + (curr_loc.y - last_loc.y)**2)
            total_dist_covered += step_dist
            last_loc = curr_loc

            # Dashboard UI
            speed = get_speed(vehicle)
            # Screen-la clear-aa dashboard vara Location offset fix pannirukkaen
            world.debug.draw_string(curr_loc + carla.Location(z=5), 
                                    f"SPEED: {int(speed)} km/h | PROGRESS: {int(total_dist_covered)}m / 6946m", 
                                    life_time=0.06, color=carla.Color(0, 255, 0))

        # 6. COMPLETION
        for _ in range(200):
            world.tick()
            spectator.set_transform(camera.get_transform())
            world.debug.draw_string(vehicle.get_location() + carla.Location(z=6), 
                                    "LOOP COMPLETED!", color=carla.Color(255, 0, 0), life_time=0.1)

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if 'vehicle' in locals(): vehicle.destroy()
        if 'camera' in locals(): camera.destroy()
        print("Done!")

if __name__ == '__main__':
    main()
