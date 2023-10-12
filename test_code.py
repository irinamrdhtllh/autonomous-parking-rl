from datetime import datetime

import gymnasium as gym
import imageio
import numpy as np

import carla_integration


def concat(img):
    # Gray scale
    # img1 = img[:, :, 0]
    # img2 = img[:, :, 1]
    # img3 = img[:, :, 2]
    # img4 = img[:, :, 3]

    # RGB
    img1 = img[:, :, :3]
    img2 = img[:, :, 3:6]
    img3 = img[:, :, 6:9]
    img4 = img[:, :, 9:]

    top_row = np.concatenate([img1, img2], axis=1)
    bottom_row = np.concatenate([img3, img4], axis=1)
    result = np.concatenate([top_row, bottom_row], axis=0)
    return result


def main():
    env = gym.make("Carla-v1")
    # images = []
    try:
        observation, info = env.reset()
        done = False
        while not done:
            # images.append(concat(observation["image"]))
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        # now = datetime.now().strftime("%Y%m%d_%H%M%S")
        # imageio.mimsave(f"Carla-v1_{now}.gif", images, duration=30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
