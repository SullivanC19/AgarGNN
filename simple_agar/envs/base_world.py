import gymnasium as gym
from gymnasium import spaces

import numpy as np
from scipy.spatial import distance_matrix


class BaseWorld(gym.Env):
    def __init__(
        self,
        num_players: int = 1,
        num_pellets: int = 200,
        pellet_mass: float = 1.0,
        player_mass_base: float = 10.0,
        player_mass_decay: float = 0.999,
        player_speed_inv_pow: float = -0.44,
        player_speed_scale: float = 10.0,
        world_size: int = 500,
        sqrt_mass_to_radius: float = 1.0,
    ):

        self.num_players = num_players
        self.num_pellets = num_pellets
        self.pellet_mass = pellet_mass
        self.pellet_radius = np.sqrt(self.pellet_mass) * sqrt_mass_to_radius
        self.player_mass_base = player_mass_base
        self.player_mass_decay = player_mass_decay
        self.player_speed_inv_pow = player_speed_inv_pow
        self.player_speed_scale = player_speed_scale
        self.world_size = world_size
        self.sqrt_mass_to_radius = sqrt_mass_to_radius

        # noop, right, up, left, down for each player
        self.action_space = spaces.MultiDiscrete(np.array([num_players, 5]))
        self._action_to_direction = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        # players and pellet positions
        self.observation_space = spaces.Dict(
            {
                "player_masses": spaces.Box(
                    low=0, high=np.inf, shape=(num_players,), dtype=np.float64
                ),
                "player_locations": spaces.Box(
                    low=0, high=world_size, shape=(num_players, 2), dtype=np.float64
                ),
                "pellet_locations": spaces.Box(
                    low=0, high=world_size, shape=(num_pellets, 2), dtype=np.float64
                ),
            }
        )

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._player_masses = np.full(
            self.num_players, self.player_mass_base, dtype=np.float64
        )
        self._player_locations = (
            self.np_random.random((self.num_players, 2)) * self.world_size
        )
        self._pellet_locations = (
            self.np_random.random((self.num_pellets, 2)) * self.world_size
        )

        self._update_player_distances()
        self._update_player_radii()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, actions):
        self._update_player_locations(actions)
        self._update_player_distances()
        self._update_player_radii()

        prev_masses = self._player_masses.copy()
        _, pellets_eaten = self._update_player_masses()
        self._update_pellet_locations(pellets_eaten)

        observation = self._get_observation()
        reward = self._get_reward(prev_masses)
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        # TODO @macnew implement this
        ...

    def close(self):
        # TODO @macnew implement this
        ...

    def _update_player_locations(self, actions):
        self._player_locations = np.clip(
            self._player_locations
            + self._action_to_direction[actions]
            * np.power(self._player_masses, self.player_speed_inv_pow)
            * self.player_speed_scale,
            0,
            self.world_size,
        )

    def _update_player_radii(self):
        self._player_radii = np.sqrt(self._player_masses) * self.sqrt_mass_to_radius

    def _update_player_distances(self):
        self._player_to_player_distances = distance_matrix(
            self._player_locations, self._player_locations
        )
        self._player_to_pellet_distances = distance_matrix(
            self._player_locations, self._pellet_locations
        )

    def _update_player_masses(self):
        # add mass for eaten pellets and players
        RR = np.add.outer(self._player_radii, self._player_radii)
        player_eats_player = (
            (self._player_to_player_distances < RR)
            & (self._player_radii[:, np.newaxis] > self._player_radii[np.newaxis, :])
        ).astype(np.float64)
        player_eats_pellet = (
            (self._player_to_pellet_distances < self.pellet_radius)
            & (self._player_radii[:, np.newaxis] > self.pellet_radius)
        ).astype(np.float64)

        players_eaten = np.any(player_eats_player, axis=0)
        pellets_eaten = np.any(player_eats_pellet, axis=0)

        # divide mass evenly among players that ate the same player or pellet
        player_eats_player[:, players_eaten] /= np.sum(
            player_eats_player[:, players_eaten], axis=1, keepdims=True
        )
        player_eats_pellet[:, pellets_eaten] /= np.sum(
            player_eats_pellet[:, pellets_eaten], axis=1, keepdims=True
        )
        self._player_masses += np.sum(player_eats_pellet * self.pellet_mass, axis=1)
        self._player_masses += np.sum(player_eats_player * self._player_masses, axis=1)

        # decay player mass
        self._player_masses *= self.player_mass_decay

        # set eaten players to 0 mass
        self._player_masses = np.where(players_eaten, 0, self._player_masses)

        return players_eaten, pellets_eaten

    def _update_pellet_locations(self, pellets_eaten):
        # respawn eaten pellets in random unoccupied locations
        for i in np.where(pellets_eaten)[0]:
            while np.any(
                np.linalg.norm(
                    self._pellet_locations[i] - self._player_locations, axis=-1
                )
                < self._player_radii
            ):
                self._pellet_locations[i] = self.np_random.rand(2) * self.world_size

    def _get_observation(self):
        return {
            "player_masses": self._player_masses,
            "player_locations": self._player_locations,
            "pellet_locations": self._pellet_locations,
        }

    def _get_reward(self, prev_masses):
        return self._player_masses - prev_masses

    def _get_terminated(self):
        return np.count_nonzero(self._player_masses) <= 1

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {
            "player_radii": self._player_radii,
            "player_to_player_distances": self._player_to_player_distances,
            "player_to_pellet_distances": self._player_to_pellet_distances,
        }
