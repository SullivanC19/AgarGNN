import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import Env

from simple_agar.wrappers.single_player import SinglePlayer

from constants import HIDDEN_LAYERS, HIDDEN_SIZE, NEGATIVE_SLOPE, K_PLAYER, K_PELLET


class MLPModel(nn.Module):
    f"""A multi-layer-perceptron (MLP) model for the simple agar environment.

    Baseline model that uses an MLP to model the policy of an agent in the simple agar environment.
    The model takes as input the observation of the environment and outputs a log probability
    distribution over the action space of the agent.

    Parameters
    ----------
    env : Env
        The environment for which the model is being constructed.

    Keyword Arguments
    -----------------
    k_player : int
        The number of nearest players to consider in the observation. (default: {K_PLAYER})
    k_pellet : int
        The number of nearest pellets to consider in the observation. (default: {K_PELLET})
    hidden_layers : int
        The number of hidden layers in the MLP model. (default: {HIDDEN_LAYERS})
    hidden_size : int
        The number of neurons in each hidden layer. (default: {HIDDEN_SIZE})
    negative_slope : float
        The negative slope of the leaky ReLU activation function. (default: {NEGATIVE_SLOPE})
    """

    def __init__(
        self,
        env: Env,
        k_player: int = K_PLAYER,
        k_pellet: int = K_PELLET,
        hidden_layers: int = HIDDEN_LAYERS,
        hidden_size: int = HIDDEN_SIZE,
        negative_slope: float = NEGATIVE_SLOPE,
    ):
        super().__init__()

        # MLP should only be used for single-player agario environments
        is_single_player = False
        inner_env = env
        while hasattr(inner_env, "env"):
            if isinstance(inner_env, SinglePlayer):
                is_single_player = True
                break
            inner_env = inner_env.env

        if not is_single_player:
            raise ValueError(
                "MLPModel should only be used for single-player agario environment"
            )

        self._env = env

        self._k_player = (
            env.num_players if k_player == -1 else min(k_player, env.num_players)
        )
        self._k_pellet = (
            env.num_pellets if k_pellet == -1 else min(k_pellet, env.num_pellets)
        )

        input_size = self._k_player * 3 + self._k_pellet * 2
        output_size = env.action_space.n

        self._lin = nn.ModuleList(
            [
                nn.Linear(
                    input_size if l == 0 else hidden_size,
                    hidden_size if l < hidden_layers else output_size,
                )
                for l in range(hidden_layers + 1)
            ]
        )

        self._negative_slope = negative_slope

    def forward(self, observation, info):
        # get top k_player players and k_pellet pellets in order of distance to player
        player_ordering = np.argsort(
            info["player_to_player_distances"][self._env.player_idx]
        )
        pellet_ordering = np.argsort(
            info["player_to_pellet_distances"][self._env.player_idx]
        )

        x = torch.from_numpy(
            np.hstack(
                [
                    observation["player_masses"][player_ordering][: self._k_player],
                    observation["player_locations"][player_ordering][
                        : self._k_player
                    ].flatten(),
                    observation["pellet_locations"][pellet_ordering][
                        : self._k_pellet
                    ].flatten(),
                ],
            ).reshape((1, -1))
        ).float()

        # feed through network
        for i in range(len(self._lin) - 1):
            x = F.leaky_relu(self._lin[i](x), negative_slope=self._negative_slope)

        x = self._lin[-1](x)
        x = F.log_softmax(x, dim=-1)
        return x
