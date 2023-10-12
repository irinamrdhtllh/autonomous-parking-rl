import datetime
import math

import carla
import imageio
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, Tuple

from experiment.base_experiment import BaseExperiment
from helper.carla_helper import post_process_image


class Experiment(BaseExperiment):
    def __init__(self, exp_config):
        super().__init__(exp_config)

        self.framestack = self.exp_config["framestack"]
        self.max_time_idle = self.exp_config["max_time_idle"]
        self.max_time_episode = self.exp_config["max_time_episode"]

        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

        self.last_action = None

    def reset(self):
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.collision_impulse = 0
        self.hero_collided = False
        self.hero_parked = False

        self.done_time_idle = False
        self.done_time_episode = False
        self.done_collision = False
        self.done_parked = False

        self.terminated = False
        self.truncated = False

        # Hero variables
        self.last_location = None
        self.last_goal_distance = None
        self.last_velocity = 0
        self.goal_location = None

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

    def get_action_space(self):
        if self.exp_config["continuous"]:
            return Tuple(
                Box(
                    low=np.array([0.0, -0.75, 0.0]),
                    high=np.array([0.75, 0.75, 0.75]),
                    dtype=np.float32,
                ),
                Discrete(2),
                Discrete(2),
            )
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        num_channels = 1 if self.exp_config["hero"]["camera_grayscale"] else 3
        num_cameras = 4
        image_space = Box(
            low=0,
            high=255,
            shape=(
                self.exp_config["hero"]["sensors"]["front_cam"]["image_size_x"],
                self.exp_config["hero"]["sensors"]["front_cam"]["image_size_y"],
                num_cameras * num_channels,
            ),
            dtype=np.uint8,
        )
        distance_space = Box(
            low=-1,
            high=self.exp_config["hero"]["sensors"]["lidar"]["range"],
            shape=(self.exp_config["hero"]["max_lidar_actors"],),
            dtype=np.float32,
        )

        obs_space = Dict({"image": image_space, "obj_distance": distance_space})

        return obs_space

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 0.25, False, False],  # Apply Break
            2: [0.0, 0.00, 0.5, False, False],  # Apply Break
            3: [0.0, 0.00, 0.75, False, False],  # Apply Break
            4: [0.0, 0.75, 0.0, False, False],  # Right
            5: [0.0, 0.50, 0.0, False, False],  # Right
            6: [0.0, 0.25, 0.0, False, False],  # Right
            7: [0.0, -0.75, 0.0, False, False],  # Left
            8: [0.0, -0.50, 0.0, False, False],  # Left
            9: [0.0, -0.25, 0.0, False, False],  # Left
            10: [0.25, 0.00, 0.0, False, False],  # Straight
            11: [0.25, 0.75, 0.0, False, False],  # Right
            12: [0.25, 0.50, 0.0, False, False],  # Right
            13: [0.25, 0.25, 0.0, False, False],  # Right
            14: [0.25, -0.75, 0.0, False, False],  # Left
            15: [0.25, -0.50, 0.0, False, False],  # Left
            16: [0.25, -0.25, 0.0, False, False],  # Left
            17: [0.5, 0.00, 0.0, False, False],  # Straight
            18: [0.5, 0.75, 0.0, False, False],  # Right
            19: [0.5, 0.50, 0.0, False, False],  # Right
            20: [0.5, 0.25, 0.0, False, False],  # Right
            21: [0.5, -0.75, 0.0, False, False],  # Left
            22: [0.5, -0.50, 0.0, False, False],  # Left
            23: [0.5, -0.25, 0.0, False, False],  # Left
            24: [0.75, 0.00, 0.0, False, False],  # Straight
            25: [0.75, 0.75, 0.0, False, False],  # Right
            26: [0.75, 0.50, 0.0, False, False],  # Right
            27: [0.75, 0.25, 0.0, False, False],  # Right
            28: [0.75, -0.75, 0.0, False, False],  # Left
            29: [0.75, -0.50, 0.0, False, False],  # Left
            30: [0.75, -0.25, 0.0, False, False],  # Left
            31: [0.0, 0.00, 0.25, False, True],  # Apply Break (Reverse)
            32: [0.0, 0.00, 0.5, False, True],  # Apply Break (Reverse)
            33: [0.0, 0.00, 0.75, False, True],  # Apply Break (Reverse)
            34: [0.0, 0.75, 0.0, False, True],  # Right (Reverse)
            35: [0.0, 0.50, 0.0, False, True],  # Right (Reverse)
            36: [0.0, 0.25, 0.0, False, True],  # Right (Reverse)
            37: [0.0, -0.75, 0.0, False, True],  # Left (Reverse)
            38: [0.0, -0.50, 0.0, False, True],  # Left (Reverse)
            39: [0.0, -0.25, 0.0, False, True],  # Left (Reverse)
            40: [0.25, 0.00, 0.0, False, True],  # Straight (Reverse)
            41: [0.25, 0.75, 0.0, False, True],  # Right (Reverse)
            42: [0.25, 0.50, 0.0, False, True],  # Right (Reverse)
            43: [0.25, 0.25, 0.0, False, True],  # Right (Reverse)
            44: [0.25, -0.75, 0.0, False, True],  # Left (Reverse)
            45: [0.25, -0.50, 0.0, False, True],  # Left (Reverse)
            46: [0.25, -0.25, 0.0, False, True],  # Left (Reverse)
            47: [0.5, 0.00, 0.0, False, True],  # Straight (Reverse)
            48: [0.5, 0.75, 0.0, False, True],  # Right (Reverse)
            49: [0.5, 0.50, 0.0, False, True],  # Right (Reverse)
            50: [0.5, 0.25, 0.0, False, True],  # Right (Reverse)
            51: [0.5, -0.75, 0.0, False, True],  # Left (Reverse)
            52: [0.5, -0.50, 0.0, False, True],  # Left (Reverse)
            53: [0.5, -0.25, 0.0, False, True],  # Left (Reverse)
            54: [0.75, 0.00, 0.0, False, True],  # Straight (Reverse)
            55: [0.75, 0.75, 0.0, False, True],  # Right (Reverse)
            56: [0.75, 0.50, 0.0, False, True],  # Right (Reverse)
            57: [0.75, 0.25, 0.0, False, True],  # Right (Reverse)
            58: [0.75, -0.75, 0.0, False, True],  # Left (Reverse)
            59: [0.75, -0.50, 0.0, False, True],  # Left (Reverse)
            60: [0.75, -0.25, 0.0, False, True],  # Left (Reverse)
        }

    def compute_action(self, action):
        vehicle_control = carla.VehicleControl()

        if self.exp_config["continuous"]:
            throttle = action[0].item()
            steer = action[1].item()
            brake = action[2].item()

            if throttle > brake:
                vehicle_control.throttle = throttle
                vehicle_control.brake = 0
            elif throttle < brake:
                vehicle_control.throttle = 0
                vehicle_control.brake = brake
            vehicle_control.steer = steer
            vehicle_control.reverse = False
            vehicle_control.hand_brake = False

        else:
            action_control = self.get_actions()[int(action)]

            vehicle_control.throttle = action_control[0]
            vehicle_control.steer = action_control[1]
            vehicle_control.brake = action_control[2]
            vehicle_control.hand_brake = action_control[3]
            vehicle_control.reverse = action_control[4]

        self.last_action = vehicle_control

        return vehicle_control

    def get_observation(self, core, sensor_data):
        world = core.world
        hero = core.hero
        hero_location = hero.get_location()

        collision = sensor_data.get("collision")
        if collision is not None:
            self.hero_collided = True
            self.collision_impulse = sensor_data["collision"][1][1]
            collision_data = [
                "Object: " + str(sensor_data["collision"][1][0]),
                "Intensity: " + str(sensor_data["collision"][1][1]),
            ]
            with open("collision_history.txt", "a") as f:
                f.writelines("\n".join(collision_data))
                f.write("\n")

        lidar_data = sensor_data["lidar"][1]
        lidar_actor_idx = lidar_data["ObjIdx"]
        actor_idx = []
        for id in lidar_actor_idx:
            if id not in actor_idx and id != hero.id:
                actor_idx.append(id)

        world_actors = world.get_actors()
        lidar_actors = []
        for id in actor_idx:
            if id != 0:
                actor = world_actors.find(int(id))
                lidar_actors.append(actor)

        actor_distance_list = []
        for actor in lidar_actors:
            if actor is not None:
                actor_location = actor.get_location()
                actor_distance = float(
                    np.sqrt(
                        np.square(hero_location.x - actor_location.x)
                        + np.square(hero_location.y - actor_location.y)
                    )
                )
                actor_distance_list.append(actor_distance)

        max_lidar_actors = self.exp_config["hero"]["max_lidar_actors"]
        if len(actor_distance_list) < max_lidar_actors:
            actor_distance_list += [-1] * (max_lidar_actors - len(actor_distance_list))
        else:
            actor_distance_list = sorted(actor_distance_list)[:max_lidar_actors]

        actor_distance_list = np.array(actor_distance_list, dtype=np.float32)

        stacked_image = None
        for key in sensor_data.keys():
            if "cam" not in key:
                continue

            id, sensor_reading = sensor_data[key]
            image = post_process_image(
                sensor_reading,
                normalized=self.exp_config["hero"]["camera_normalized"],
                grayscale=self.exp_config["hero"]["camera_grayscale"],
            )
            if stacked_image is None:
                stacked_image = image
            else:
                stacked_image = np.dstack([stacked_image, image])

        return {"image": stacked_image, "obj_distance": actor_distance_list}, {}

    def get_done_status(self, observation, core):
        hero = core.hero
        hero_velocity = hero.get_velocity()
        hero_velocity = 3.6 * math.sqrt(
            hero_velocity.x**2 + hero_velocity.y**2 + hero_velocity.z**2
        )

        if self.hero_collided:
            self.done_collision = True

        self.hero_parked = core.parked
        if self.hero_parked:
            self.done_parked = True

        if hero_velocity > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.done_time_idle = self.max_time_idle < self.time_idle

        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode

        if self.done_collision or self.done_parked:
            self.terminated = True
        if self.done_time_episode or self.done_time_idle:
            self.truncated = True

        return self.truncated, self.terminated

    def compute_reward(self, observation, core):
        hero = core.hero
        reward = 0

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = hero.get_velocity()
        hero_velocity = 3.6 * math.sqrt(
            hero_velocity.x**2 + hero_velocity.y**2 + hero_velocity.z**2
        )
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Initialize last location
        if self.last_location is None:
            self.last_location = hero_location
        self.goal_location = core.goal_location

        # Distance to goal
        goal_distance = float(
            np.sqrt(
                np.square(hero_location.x - self.goal_location.x)
                + np.square(hero_location.y - self.goal_location.y)
            )
        )
        if self.last_goal_distance is None:
            self.last_goal_distance = goal_distance

        delta_goal_distance = self.last_goal_distance - goal_distance
        delta_velocity = hero_velocity - self.last_velocity

        # Update variables
        self.last_location = hero_location
        self.last_goal_distance = goal_distance
        self.last_velocity = hero_velocity

        # Reward if going closer to goal
        reward += 1 * delta_goal_distance

        # Reward if going faster that last step
        if hero_velocity < 30:
            reward += 0.5 * delta_velocity
        elif hero_velocity > 40:
            reward += -1 * delta_velocity

        if self.done_time_idle:
            print("Done idle.")
            reward += -100
        if self.done_time_episode:
            print("Done max time.")
            reward += -100
        if self.done_collision:
            print("Done collided with other object.")
            reward += -0.1 * self.collision_impulse
        if self.done_parked:
            print("Hero successfully parked!")
            reward += 100

        return reward
