import numpy as np

from gymnasium import Wrapper
from gymnasium.spaces import Discrete
from typing import SupportsInt

from simple_agar.envs.base_world import BaseWorld
from simple_agar.agents.greedy_agent import get_greedy_player_actions

from constants import WINDOW_SIZE, FPS
    

class SinglePlayer(Wrapper):
    def __init__(self, env: BaseWorld, player_idx: int):
        super().__init__(env)
        self.player_idx = player_idx
        self.action_space = Discrete(env.action_space.nvec[player_idx])

    def step(self, action):
        # all other players take greedy actions
        all_actions = get_greedy_player_actions(
            self.env._player_to_pellet_distances,
            self.env._pellet_locations,
            self.env._player_locations,
        )
        all_actions[self.player_idx] = action[self.player_idx]

        observation, reward, terminated, truncated, info = self.env.step(all_actions)
        terminated |= not self.env._player_is_alive[self.player_idx]
        reward = reward[np.newaxis, self.player_idx]
        
        return observation, reward, terminated, truncated, info

    def render(self, window_size=WINDOW_SIZE, fps=FPS):
        self.env.render(window_size=window_size, fps=fps, highlight_player_index=self.player_idx)
