import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import Env

from simple_agar.wrappers.single_player import SinglePlayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Module):
    def __init__(
            self,
            env: Env,
            hidden_layers: int = 3,
            hidden_size: int = 64,
            k_players: int = -1,
            k_pellets: int = -1,
            negative_slope: float = 0.2):
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

        num_players = env.num_players if k_players == -1 else min(k_players, env.num_players)
        num_pellets = env.num_pellets if k_pellets == -1 else min(k_pellets, env.num_pellets)

        self.k_players = num_players
        self.k_pellets = num_pellets

        input_size = self.k_players * 3 + self.k_pellets * 2
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
                observation["player_masses"][player_ordering][:self.k_players],
                observation["player_locations"][player_ordering][:self.k_players].flatten(),
                observation["pellet_locations"][pellet_ordering][:self.k_pellets].flatten(),
            ],
        )).to(device).float()

        # feed through network
        for i in range(len(self.lin) - 1):
            x = F.leaky_relu(self.lin[i](x), negative_slope=self.negative_slope)
        
        x = self.lin[-1](x)
        x = F.log_softmax(x)
        return x