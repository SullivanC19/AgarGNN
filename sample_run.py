import gymnasium as gym
import numpy as np

import simple_agar

if __name__ == "__main__":
    config = {
        "num_players": 2,
        "num_pellets": 1,
        "render_mode": "human",
    }

    env = gym.make("simple_agar/BaseWorld-v0", **config)
    print(env.reset())

    for i in range(10):
        print(env.step(np.array([1, 1])))
