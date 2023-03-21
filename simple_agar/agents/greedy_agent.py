import numpy as np

from simple_agar.agents.base_agent import BaseAgent


def get_greedy_player_actions(
    player_to_pellet_distances: np.ndarray,
    pellet_locations: np.ndarray,
    player_locations: np.ndarray,
    player_indices: np.ndarray = None,
):
    """Return the actions that would move each player towards the closest pellet.

    Parameters
    ----------
    player_to_pellet_distances : np.ndarray, shape (n_players, n_pellets)
        The distances from each player to each pellet.
    pellet_locations : np.ndarray, shape (n_pellets, 2)
        The locations of each pellet.
    player_locations : np.ndarray, shape (n_players, 2)
        The locations of each player.
    player_indices : np.ndarray, optional, shape (n_players,)
        The indices of the players to get actions for. If None, all players are
        considered.

    Returns
    -------
    actions : np.ndarray, shape (n_players,)
        The actions that would move each player towards the closest pellet.
    """

    if player_indices is not None:
        player_to_pellet_distances = player_to_pellet_distances[player_indices]
        player_locations = player_locations[player_indices]

    closest_pellets = np.argmin(player_to_pellet_distances, axis=1)
    closest_pellet_locations = pellet_locations[closest_pellets]

    left_dist = closest_pellet_locations[:, 0] - player_locations[:, 0]
    right_dist = player_locations[:, 0] - closest_pellet_locations[:, 0]
    down_dist = closest_pellet_locations[:, 1] - player_locations[:, 1]
    up_dist = player_locations[:, 1] - closest_pellet_locations[:, 1]

    actions = 1 + np.argmin(
        np.stack([right_dist, up_dist, left_dist, down_dist]), axis=0
    )

    return actions


class GreedyAgent(BaseAgent):
    """An agent that moves towards the closest pellet."""

    def __init__(self, player_idx=0):
        super().__init__()
        self.player_idx = player_idx

    def act(self, observation, info):
        action = get_greedy_player_actions(
            info["player_to_pellet_distances"],
            observation["pellet_locations"],
            observation["player_locations"],
            player_indices=[self.player_idx],
        )
        return action, 0.0
