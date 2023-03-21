import numpy as np

from gymnasium import Wrapper, Env
from gymnasium.spaces import Discrete
from typing import SupportsInt

from simple_agar.agents.greedy_agent import get_greedy_player_actions

from constants import WINDOW_SIZE, FPS


class SinglePlayer(Wrapper):
    """A wrapper that makes the environment single-player by taking greedy actions
    for all other players.

    Parameters
    ----------
    env : Env
        The simple agar environment to wrap.
    player_idx : int
        The index of the player to control.
    """

    def __init__(self, env: Env, player_idx: int):
        super().__init__(env)
        self.player_idx = player_idx
        self.action_space = Discrete(env.action_space.nvec[player_idx])

    def step(self, action: SupportsInt):
        # all other players take greedy actions
        all_actions = get_greedy_player_actions(
            self.env._player_to_pellet_distances,
            self.env._pellet_locations,
            self.env._player_locations,
        )
        all_actions[self.player_idx] = action

        observation, reward, terminated, truncated, info = self.env.step(all_actions)

        # terminate early if the controlled player is dead
        terminated |= not self.env._player_is_alive[self.player_idx]

        # only return the reward for the controlled player
        reward = reward[np.newaxis, self.player_idx]

        return observation, reward, terminated, truncated, info

    def render(self, window_size=WINDOW_SIZE, fps=FPS):
        self.env.render(
            window_size=window_size, fps=fps, highlight_player_index=self.player_idx
        )
