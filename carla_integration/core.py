import logging
import os
import random
import subprocess
import time

import carla
import numpy as np

from helper.carla_helper import is_used
from helper.sensors.sensor_factory import SensorFactory
from helper.sensors.sensor_interface import SensorInterface


class CarlaCore:
    def __init__(self, carla_config, exp_config):
        self.carla_config = carla_config
        self.exp_config = exp_config

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None

        self.hero = None
        self.spawn_point = None
        self.set_point = None
        self.parked = False
        self.goal_boundary = None
        self.goal_location = None

        self.sensor_interface = SensorInterface()

        self.mode = self.exp_config["mode"]
        self.scenario = self.exp_config["scenario"]

        self.parked_cars = []
        self.parked_cars_id = []

        self.walkers = []
        self.all_id = []
        self.all_walkers = []

        self.init_server()
        self.connect_client()

    def init_server(self):
        self.server_port = random.randint(15000, 32000)

        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + str(self.server_port))
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.carla_config["show_display"]:
            server_command = [
                "{}\CarlaUE4.exe".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.carla_config["resolution_x"]),
                "-ResY={}".format(self.carla_config["resolution_y"]),
                "-quality-level={}".format(self.carla_config["quality_level"]),
            ]
        else:
            server_command = [
                "{}\CarlaUE4.exe".format(os.environ["CARLA_ROOT"]),
                "-RenderOffScreen",
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        for i in range(self.carla_config["retries_on_error"]):
            try:
                self.client = carla.Client(self.carla_config["host"], self.server_port)
                self.client.set_timeout(self.carla_config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.carla_config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.carla_config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return
            except Exception as e:
                print(
                    " Waiting for server to be ready: {}, attempt {} of {}".format(
                        e, i + 1, self.carla_config["retries_on_error"]
                    )
                )
                time.sleep(5)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration"
        )

    def setup_experiment(self):
        self.world = self.client.load_world(
            map_name=self.exp_config["town"],
            reset_settings=False,
            map_layers=carla.MapLayer.All
            if self.carla_config["enable_map_assets"]
            else carla.MapLayer.NONE,
        )

        self.map = self.world.get_map()

        weather = getattr(carla.WeatherParameters, self.exp_config["weather"])
        self.world.set_weather(weather)

        self.tm_port = 8000
        while is_used(self.tm_port):
            print(
                "Traffic manager's port "
                + str(self.tm_port)
                + " is already being used. Checking the next one"
            )
            self.tm_port += 1
        print("Traffic manager connected to port " + str(self.tm_port))

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_hybrid_physics_mode(
            self.exp_config["background_activity"]["tm_hybrid_mode"]
        )

        seed = self.exp_config["background_activity"]["seed"]
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

        # Draw the goal boundary
        # top_left = eval(self.exp_config["hero"]["goal_boundary"]["top_left"])
        # top_right = eval(self.exp_config["hero"]["goal_boundary"]["top_right"])
        # bottom_left = eval(self.exp_config["hero"]["goal_boundary"]["bottom_left"])
        # bottom_right = eval(self.exp_config["hero"]["goal_boundary"]["bottom_right"])
        # center = eval(self.exp_config["hero"]["goal_boundary"]["center"])

        # self.goal_boundary = [top_left, top_right, bottom_left, bottom_right, center]

        # for point in self.goal_boundary:
        #     self.world.debug.draw_point(point, size=0.05, life_time=0)

        top_left = eval(self.exp_config["hero"]["goal_boundary"]["top_left"])
        top_right = eval(self.exp_config["hero"]["goal_boundary"]["top_right"])
        bottom_left = eval(self.exp_config["hero"]["goal_boundary"]["bottom_left"])
        bottom_right = eval(self.exp_config["hero"]["goal_boundary"]["bottom_right"])
        center = eval(self.exp_config["hero"]["goal_boundary"]["center"])

        self.goal_location = center

        extent_x = (top_left.x - top_right.x) / 2
        extent_y = (top_left.y - bottom_left.y) / 2
        extent_z = 0

        self.world.debug.draw_box(
            carla.BoundingBox(
                center,
                carla.Vector3D(extent_x, extent_y, extent_z),
            ),
            carla.Rotation(pitch=0, yaw=0, roll=0),
            thickness=0.2,
            color=carla.Color(r=255, g=0, b=0),
            life_time=0,
        )

    def spawn_hero(self):
        # Destroy hero if not NONE
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        # Destroy all hero's sensors
        self.sensor_interface.destroy()

        self.world.tick()

        spawn_point_loc = eval(self.exp_config["hero"]["spawn_point_loc"])
        spawn_point_rot = eval(self.exp_config["hero"]["spawn_point_rot"])
        hero_spawn_point = carla.Transform(spawn_point_loc, spawn_point_rot)

        hero_model = "".join(self.exp_config["hero"]["model"])
        hero_blueprint = self.world.get_blueprint_library().find(hero_model)
        hero_blueprint.set_attribute("role_name", "hero")

        self.hero = self.world.spawn_actor(hero_blueprint, hero_spawn_point)
        if self.hero is None:
            raise AssertionError(
                f"Error spawning hero: {hero_blueprint} at point {hero_spawn_point} index {self.hero_spawn_point_id}"
            )

        self.world.tick()

        if self.hero is not None:
            print("Hero spawned!")
            for name, attributes in self.exp_config["hero"]["sensors"].items():
                sensor = SensorFactory.spawn(
                    name, attributes, self.sensor_interface, self.hero
                )

    def hero_parked(self):
        # Hero's bounding box and transform
        hero_bounding_box = self.hero.bounding_box
        extent_x = hero_bounding_box.extent.x
        extent_y = hero_bounding_box.extent.y
        hero_location = hero_bounding_box.location
        hero_rotation = hero_bounding_box.rotation

        left_x = hero_location.x - extent_x
        right_x = hero_location.x + extent_x
        top_y = hero_location.y - extent_y
        bottom_y = hero_location.y + extent_y

        # Goal's bounding box
        top_left = self.goal_boundary[0]
        top_right = self.goal_boundary[1]
        bottom_left = self.goal_boundary[2]
        bottom_right = self.goal_boundary[3]
        center = self.goal_boundary[4]

        # Check if the hero is parked
        if (
            hero_location.x <= (center.x - 0.25) or hero_location.x >= (center.x + 0.25)
        ) and (
            hero_location.y <= (center.y - 0.25) or hero_location.y >= (center.y + 0.25)
        ):
            if (
                (left_x >= top_left.x)
                and (right_x <= bottom_right.x)
                and (top_y >= top_right.y)
                and (bottom_y <= bottom_left.y)
            ):
                self.parked = True

    def tick(self, control):
        # Move hero car
        if control is None:
            pass
        else:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.carla_config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        transform = self.hero.get_transform()

        # Back-view camera
        # Get the camera position
        # server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        # server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        # server_view_z = transform.location.z + 3

        # Get the camera orientation
        # server_view_roll = transform.rotation.roll
        # server_view_yaw = transform.rotation.yaw
        # server_view_pitch = transform.rotation.pitch

        # Bird-view camera
        # Set the camera position
        server_view_x = -10
        server_view_y = -31
        server_view_z = 30

        # Set the camera orientation
        server_view_roll = 0
        server_view_yaw = -90
        server_view_pitch = -90

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(
                    pitch=server_view_pitch, yaw=server_view_yaw, roll=server_view_roll
                ),
            )
        )

    def get_sensor_data(self):
        sensor_data = self.sensor_interface.get_data()
        return sensor_data

    def apply_hero_control(self, control):
        self.hero.apply_control(control)

    def spawn_parked_cars(self):
        parking_points = self.exp_config["background_activity"]["parking_points"]
        parking_points.remove(self.exp_config["hero"]["goal_boundary"]["center"])
        for point in parking_points:
            point = eval(point)
            self.world.debug.draw_point(
                point, size=0.05, color=carla.Color(r=0, g=255, b=0), life_time=0
            )

    def spawn_walkers(self):
        walker_blueprints = self.world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )

        n_walkers = random.choice(self.exp_config["background_activity"]["n_walkers"])

        percentage_walker_running = 0.0
        percentage_walker_crossing = 0.0

        SpawnActor = carla.command.SpawnActor

        spawn_points = []
        for i in range(n_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_blueprint = random.choice(walker_blueprints)
            if walker_blueprint.has_attribute("is_invicible"):
                walker_blueprint.set_attribute("is_invicible", "false")
            if walker_blueprint.has_attribute("speed"):
                if random.random() > percentage_walker_running:
                    walker_speed.append(
                        walker_blueprint.get_attribute("speed").recommended_values[1]
                    )
                else:
                    walker_speed.append(
                        walker_blueprint.get_attribute("speed").recommended_values[2]
                    )
            else:
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_blueprint, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed_dummy = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers.append({"id": results[i].actor_id})
                walker_speed_dummy.append(walker_speed[i])
        walker_speed = walker_speed_dummy

        batch = []
        controller_blueprint = self.world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for i in range(len(self.walkers)):
            batch.append(
                SpawnActor(
                    controller_blueprint, carla.Transform(), self.walkers[i]["id"]
                )
            )
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers[i]["con"] = results[i].actor_id

        for i in range(len(self.walkers)):
            self.all_id.append(self.walkers[i]["con"])
            self.all_id.append(self.walkers[i]["id"])
        self.all_walkers = self.world.get_actors(self.all_id)

        self.world.set_pedestrians_cross_factor(percentage_walker_crossing)
        for i in range(0, len(self.all_id), 2):
            self.all_walkers[i].start()
            self.all_walkers[i].go_to_location(
                self.world.get_random_location_from_navigation()
            )
            self.all_walkers[i].set_max_speed(float(walker_speed[int(i / 2)]))

    def destroy(self):
        # Destroy all actors
        if len(self.parked_cars_id) != 0:
            print("\nDestroying %d vehicles" % len(self.parked_cars_id))
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.parked_cars_id]
            )
            self.cars_list = []

        if len(self.walkers) != 0:
            for i in range(0, len(self.all_id), 2):
                self.all_walkers[i].stop()
            print("\nDestroying %d walkers" % len(self.walkers))
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.all_id]
            )
            self.walkers = []
            self.all_id = []
            self.all_walkers = []
