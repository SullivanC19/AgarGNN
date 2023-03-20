import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
import pygame

import numpy as np
from scipy.spatial import distance_matrix

from constants import WINDOW_SIZE, FPS


class BaseWorld(gym.Env):
    def __init__(
        self,
        num_players: int = 1,
        num_pellets: int = 50,
        pellet_mass: float = 1.0,
        player_mass_base: float = 4.0,
        player_mass_decay: float = 0.999,
        player_speed_inv_pow: float = -0.44,
        player_speed_scale: float = 0.03,
        sqrt_mass_to_radius: float = 0.01,
        penalty_per_step: float = 0.01
    ):

        self.num_players = num_players
        self.num_pellets = num_pellets
        self.pellet_mass = pellet_mass
        self.pellet_radius = np.sqrt(self.pellet_mass) * sqrt_mass_to_radius
        self.player_mass_base = player_mass_base
        self.player_mass_decay = player_mass_decay
        self.player_speed_inv_pow = player_speed_inv_pow
        self.player_speed_scale = player_speed_scale
        self.sqrt_mass_to_radius = sqrt_mass_to_radius
        self.penalty_per_step = penalty_per_step

        self.max_diagonal_distance = np.sqrt(2)
        self.max_player_radius = self.max_diagonal_distance
        self.max_player_mass = (self.max_player_radius / self.sqrt_mass_to_radius) ** 2

        # noop, right, up, left, down for each player
        self.action_space = MultiDiscrete([[5] * num_players])
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
        self.observation_space = Dict(
            {
                "player_masses": Box(
                    low=0, high=np.inf, shape=(num_players,), dtype=np.float64
                ),
                "player_is_alive": MultiBinary(num_players),
                "player_locations": Box(
                    low=0, high=1, shape=(num_players, 2), dtype=np.float64
                ),
                "pellet_locations": Box(
                    low=0, high=1, shape=(num_pellets, 2), dtype=np.float64
                ),
            }
        )

        # rendering setup
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._player_masses = np.full(
            self.num_players, self.player_mass_base
        )
        self._player_is_alive = np.full(self.num_players, True, dtype=np.bool)
        self._player_locations = self.np_random.random((self.num_players, 2))
        self._pellet_locations = self.np_random.random((self.num_pellets, 2))

        self._update_player_distances()
        self._update_player_radii()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, actions):
        players_off_screen = self._update_player_locations(actions)
        self._update_player_distances()
        self._update_player_radii()

        prev_masses = self._player_masses.copy()
        players_eaten, pellets_eaten = self._update_player_masses()
        self._update_player_is_alive(players_eaten, players_off_screen)
        self._update_pellet_locations(pellets_eaten)

        observation = self._get_observation()
        reward = self._get_reward(prev_masses)
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, window_size=WINDOW_SIZE, fps=FPS):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_size, window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        for i in range(self.num_players):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (
                    self._player_locations[i][0] * window_size,
                    (1 - self._player_locations[i][1]) * window_size
                ),
                self._player_radii[i] * window_size,
            )
        
        for i in range(self.num_pellets):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (
                    self._pellet_locations[i][0] * window_size,
                    (1 - self._pellet_locations[i][1]) * window_size
                ),
                self.pellet_radius * window_size,
            )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _update_player_locations(self, actions):
        self._player_locations[self._player_is_alive] = self._player_locations[self._player_is_alive] \
            + self._action_to_direction[actions[self._player_is_alive]] \
            * np.power(
                self._player_masses[self._player_is_alive], self.player_speed_inv_pow
            )[:, np.newaxis] * self.player_speed_scale
        return np.nonzero((self._player_locations < 0) | (self._player_locations >= 1))[0]

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
        player_eats_player = (
            (self._player_to_player_distances < self._player_radii[:, np.newaxis])
            & (self._player_radii[:, np.newaxis] > self._player_radii[np.newaxis, :])
        ).astype(np.float64)
        player_eats_pellet = (
            (
                self._player_to_pellet_distances
                < self._player_radii[:, np.newaxis] + self.pellet_radius
            )
            & (self._player_radii[:, np.newaxis] > self.pellet_radius)
        ).astype(np.float64)

        players_eaten = np.nonzero(np.any(player_eats_player, axis=0))[0]
        pellets_eaten = np.nonzero(np.any(player_eats_pellet, axis=0))[0]

        # divide mass evenly among players that ate the same player or pellet
        player_eats_player[:, players_eaten] /= np.sum(
            player_eats_player[:, players_eaten], axis=0, keepdims=True
        )
        player_eats_pellet[:, pellets_eaten] /= np.sum(
            player_eats_pellet[:, pellets_eaten], axis=0, keepdims=True
        )
        self._player_masses += np.sum(player_eats_pellet * self.pellet_mass, axis=1)
        self._player_masses += np.sum(player_eats_player * self._player_masses, axis=1)

        # decay player mass
        self._player_masses[self._player_is_alive] = np.maximum(
            self._player_masses[self._player_is_alive] * self.player_mass_decay,
            self.player_mass_base,
        )

        return players_eaten, pellets_eaten

    def _update_player_is_alive(self, players_eaten, players_off_screen):
        # kill players that are off screen or eaten
        self._player_is_alive[players_eaten] = False
        self._player_is_alive[players_off_screen] = False

        # set dead players to zero mass
        self._player_masses[~self._player_is_alive] = 0

    def _update_pellet_locations(self, pellets_eaten):
        # respawn eaten pellets in random unoccupied locations
        for i in pellets_eaten:
            while np.any(
                np.linalg.norm(
                    self._pellet_locations[i] - self._player_locations, axis=-1
                )
                < self._player_radii
            ):
                self._pellet_locations[i] = self.np_random.random(2)

    def _get_observation(self):
        return {
            "player_masses": self._player_masses / self.max_player_mass,
            "player_is_alive": self._player_is_alive,
            "player_locations": self._player_locations,
            "pellet_locations": self._pellet_locations,
        }

    def _get_reward(self, prev_masses):
        reward = self._player_masses - prev_masses
        reward[self._player_masses == self.player_mass_base] -= self.player_mass_base * (1 - self.player_mass_decay)
        reward -= self.penalty_per_step
        return reward

    def _get_terminated(self):
        num_players_alive = np.count_nonzero(self._player_is_alive)
        return num_players_alive == 0 or (self.num_players > 1 and num_players_alive == 1)

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {
            "player_radii": self._player_radii / self.max_player_radius,
            "player_to_player_distances": self._player_to_player_distances / self.max_diagonal_distance,
            "player_to_pellet_distances": self._player_to_pellet_distances / self.max_diagonal_distance,
        }
