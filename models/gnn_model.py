import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, HeteroData

from gymnasium import Env

from constants import HIDDEN_LAYERS, HIDDEN_SIZE, NEGATIVE_SLOPE


def construct_hetero_agar_graph(env: Env, observation) -> HeteroData:
    num_players = env.num_players
    num_pellets = env.num_pellets
    pellet_mass = env.pellet_mass / env.max_player_mass

    player_masses = torch.from_numpy(observation["player_masses"])
    player_locations = torch.from_numpy(observation["player_locations"])
    pellet_locations = torch.from_numpy(observation["pellet_locations"])
    
    data = HeteroData()

    # set node features to mass values
    data['player'].x = torch.hstack([player_masses.unsqueeze(-1), player_locations])
    data['pellet'].x = torch.hstack([torch.full((num_pellets,), pellet_mass).unsqueeze(-1), pellet_locations])

    player_idxs = torch.arange(num_players)
    pellet_idxs = torch.arange(num_pellets)

    # NOTE: currently we use all edges
    # data['player', 'to', 'pellet'].edge_index = torch.cartesian_prod(player_idxs, pellet_idxs).T
    data['player', 'to', 'player'].edge_index = torch.cartesian_prod(player_idxs, player_idxs).T
    data['pellet', 'to', 'pellet'].edge_index = torch.cartesian_prod(pellet_idxs, pellet_idxs).T
    data['pellet', 'to', 'player'].edge_index = torch.cartesian_prod(pellet_idxs, player_idxs).T

    return data


class GNNModel(torch.nn.Module):
    def __init__(
            self,
            env: Env,
            model_type: str,
            graph_type: str,
            hidden_layers: int = HIDDEN_LAYERS,
            hidden_channels: int = HIDDEN_SIZE,
            negative_slope: float = NEGATIVE_SLOPE):
        super().__init__()

        self.env = env

        self.graph_builder = self.get_graph_builder(graph_type)
        self.model = self.get_model(model_type, hidden_layers, hidden_channels, negative_slope)

    def get_graph_builder(self, graph_type):
        if graph_type == "hetero":
            return construct_hetero_agar_graph
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
    def get_model(self, model_type, hidden_layers, hidden_channels, negative_slope):
        if model_type == "hetero":
            return HeteroGNN(self.env, hidden_layers, hidden_channels, negative_slope)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, observation, info):
        data = self.graph_builder(self.env, observation)
        return self.model(data.x_dict, data.edge_index_dict)
    

class HeteroGNN(torch.nn.Module):
    def __init__(
            self,
            env: Env,
            hidden_layers: int = HIDDEN_LAYERS,
            hidden_channels: int = HIDDEN_SIZE,
            negative_slope: float = NEGATIVE_SLOPE):
        super().__init__()
    
        self.env = env
        self.negative_slope = negative_slope
        self.convs = nn.ModuleList()

        for _ in range(hidden_layers):
            self.convs.append(
                pyg_nn.HeteroConv({
                    ('player', 'to', 'player'): pyg_nn.GCNConv(-1, hidden_channels, add_self_loops=False),
                    # ('player', 'to', 'pellet'): pyg_nn.SAGEConv((-1, -1), hidden_channels, add_self_loops=False, aggr='mean'),
                    ('pellet', 'to', 'player'): pyg_nn.SAGEConv((-1, -1), hidden_channels, add_self_loops=False, aggr='mean'),
                    ('pellet', 'to', 'pellet'): pyg_nn.GCNConv(-1, hidden_channels, add_self_loops=False),
                }, aggr='sum')
            )

        output_channels = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.nvec[0]
        self.post_mp = pyg_nn.Linear(hidden_channels, output_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x, negative_slope=self.negative_slope, inplace=False) for key, x in x_dict.items()}
        
        x = self.post_mp(x_dict['player'])
        return F.log_softmax(x, dim=-1)

