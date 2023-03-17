import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import Env

from simple_agar.wrappers.single_player import SinglePlayer

class MLPModel(nn.Module):
    def __init__(
            self,
            env: Env,
            hidden_layers: int = 3,
            hidden_size: int = 64,
            k_players: int = -1,
            k_pellets: int = -1):
        super().__init__()

        # MLP should only be used for single-player agario environment
        # TODO find better way to check this
        # assert isinstance(env, SinglePlayer)
        
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

    def forward(self, observation, info):
        # get top k_player players and k_pellet pellets in order of distance to player
        player_ordering = np.argsort(info["player_to_player_distances"][self.env.player_idx])
        pellet_ordering = np.argsort(info["player_to_pellet_distances"][self.env.player_idx])

        topk_ordered_player_masses = torch.from_numpy(observation["player_masses"][player_ordering][:self.k_players])
        topk_ordered_player_locations = torch.from_numpy(observation["player_locations"][player_ordering][:self.k_players])
        topk_ordered_pellet_locations = torch.from_numpy(observation["pellet_locations"][pellet_ordering][:self.k_pellets])

        # import pdb; pdb.set_trace()

        x = torch.cat(
            [
                topk_ordered_player_masses,
                topk_ordered_player_locations.flatten(),
                topk_ordered_pellet_locations.flatten(),
            ],
        ).float()

        # feed through network
        for i in range(len(self.lin) - 1):
            x = F.relu(self.lin[i](x))
        
        x = self.lin[-1](x)


        x = F.log_softmax(x)
        return x