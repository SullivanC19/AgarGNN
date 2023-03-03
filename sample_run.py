import gymnasium as gym
import numpy as np

import simple_agar

if __name__ == "__main__":
    config = {
        "num_players": 10,
        "num_pellets": 1000,
        "render_mode": "human",
    }

    # disable warnings about unflattened observation space and reward vector
    gym.logger.set_level(40)

    env = gym.make("simple_agar/BaseWorld-v0", **config)
    env.reset()

    for i in range(1000):
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
