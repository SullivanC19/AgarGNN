import numpy as np

from simple_agar.agents.base_agent import BaseAgent


def get_greedy_player_actions(
    player_to_pellet_distances, pellet_locations, player_locations, player_indices=None
):

    if player_indices is not None:
        player_to_pellet_distances = player_to_pellet_distances[player_indices]
        player_locations = player_locations[player_indices]

    closest_pellets = np.argmin(player_to_pellet_distances, axis=1)
    closest_pellet_locations = pellet_locations[closest_pellets]

    left_dist =  closest_pellet_locations[:, 0] - player_locations[:, 0]
    right_dist = player_locations[:, 0] - closest_pellet_locations[:, 0]
    down_dist = closest_pellet_locations[:, 1] - player_locations[:, 1]
    up_dist = player_locations[:, 1] - closest_pellet_locations[:, 1]

    actions = 1 + np.argmin(
        np.stack([right_dist, up_dist, left_dist, down_dist]), axis=0
    )
    
    return actions


class GreedyAgent(BaseAgent):
    def __init__(self, player_idx=0):
        super().__init__()
        self.player_idx = player_idx

    def act(self, observation, info):
        return (get_greedy_player_actions(
            info["player_to_pellet_distances"],
            observation["pellet_locations"],
            observation["player_locations"],
            player_indices=[self.player_idx],
        )[self.player_idx], 0)
