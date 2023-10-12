from __future__ import print_function

import time
from typing import Optional

import gymnasium as gym
import numpy as np

from carla_integration.core import CarlaCore
from config import read_config
from experiment.experiment import Experiment
from helper.carla_helper import kill_server


class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        self.config = read_config()
        self.carla_config = self.config["carla"]
        self.exp_config = self.config["experiment"]

        self.core = CarlaCore(self.carla_config, self.exp_config)
        self.core.setup_experiment()

        self.experiment = Experiment(self.exp_config)
        self.scenario = self.exp_config["scenario"]

        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()
        self.render_mode = render_mode
        self.spec = None

    def reset(self, *, seed=None, options=None):
        self.experiment.reset()
        self.core.destroy()
        self.core.spawn_hero()
        self.core.spawn_parked_cars()
        self.core.spawn_walkers()

        sensor_data = self.core.tick(None)
        observation, info = self.experiment.get_observation(self.core, sensor_data)

        return observation, info

    def step(self, action):
        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)
        observation, info = self.experiment.get_observation(self.core, sensor_data)
        truncated, terminated = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        kill_server()
