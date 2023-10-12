from gymnasium import register

from config import read_config

CONFIG = read_config()

register(
    id="Carla-v1",
    entry_point="carla_integration.env:CarlaEnv",
    max_episode_steps=CONFIG["experiment"]["max_time_episode"],
    reward_threshold=300.0,
)
