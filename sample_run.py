import gymnasium as gym
import numpy as np

import simple_agar

if __name__ == "__main__":
    config = {
        "num_players": 10,
        "num_pellets": 6000,
        "render_mode": "human",
    }

    # disable warnings about unflattened observation space and reward vector
    gym.logger.set_level(40)

    env = gym.make("simple_agar/BaseWorld-v0", **config)
    observation, info = env.reset()

    terminated = truncated = False
    while not terminated and not truncated:
        # action = env.action_space.sample()

        closest_pellets = np.argmin(info["player_to_pellet_distances"], axis=1)
        closest_pellet_locations = observation["pellet_locations"][closest_pellets]
        player_locations = observation["player_locations"]

        left_val = player_locations[:, 0] - closest_pellet_locations[:, 0]
        right_val = closest_pellet_locations[:, 0] - player_locations[:, 0]
        down_val = player_locations[:, 1] - closest_pellet_locations[:, 1]
        up_val = closest_pellet_locations[:, 1] - player_locations[:, 1]

        actions = 1 + np.argmax(
            np.stack([right_val, up_val, left_val, down_val], axis=1), axis=1
        )

        observation, reward, terminated, truncated, info = env.step(actions)

    env.close()
