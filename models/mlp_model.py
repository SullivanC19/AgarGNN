import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import Env

from simple_agar.wrappers.single_player import SinglePlayer

from constants import HIDDEN_LAYERS, HIDDEN_SIZE, NEGATIVE_SLOPE, K_PLAYER, K_PELLET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Module):
    def __init__(
            self,
            env: Env,
            k_player: int = K_PLAYER,
            k_pellet: int = K_PLAYER,
            hidden_layers: int = HIDDEN_LAYERS,
            hidden_size: int = HIDDEN_SIZE,
            negative_slope: float = NEGATIVE_SLOPE):
        super().__init__()

        # MLP should only be used for single-player agario environment
        is_single_player = False
        inner_env = env
        while hasattr(inner_env, "env"):
            if isinstance(inner_env, SinglePlayer):
                is_single_player = True
                break
            inner_env = inner_env.env

        if not is_single_player:
            raise ValueError("MLPModel should only be used for single-player agario environment")
        
        self.env = env

        self.k_player = env.num_players if k_player == -1 else min(k_player, env.num_players)
        self.k_pellet = env.num_pellets if k_pellet == -1 else min(k_pellet, env.num_pellets)

        input_size = self.k_player * 3 + self.k_pellet * 2
        output_size = env.action_space.n

        self.lin = nn.ModuleList([
            nn.Linear(
                input_size if l == 0 else hidden_size,
                hidden_size if l < hidden_layers else output_size
            ) for l in range(hidden_layers + 1)])
        
        self.negative_slope = negative_slope

    def forward(self, observation, info):
        # get top k_player players and k_pellet pellets in order of distance to player
        player_ordering = np.argsort(info["player_to_player_distances"][self.env.player_idx])
        pellet_ordering = np.argsort(info["player_to_pellet_distances"][self.env.player_idx])

        x = torch.from_numpy(np.hstack(
            [
                observation["player_masses"][player_ordering][:self.k_player],
                observation["player_locations"][player_ordering][:self.k_player].flatten(),
                observation["pellet_locations"][pellet_ordering][:self.k_pellet].flatten(),
            ],
        ).reshape((1, -1))).to(device).float()

        # feed through network
        for i in range(len(self.lin) - 1):
            x = F.leaky_relu(self.lin[i](x), negative_slope=self.negative_slope)
        
        x = self.lin[-1](x)
        x = F.log_softmax(x, dim=-1)
        return x