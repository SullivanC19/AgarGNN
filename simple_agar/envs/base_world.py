import gymnasium as gym
from gymnasium.spaces import Box, Dict, MultiBinary, MultiDiscrete
import pygame

import numpy as np
from scipy.spatial import distance_matrix

from constants import WINDOW_SIZE, FPS


class BaseWorld(gym.Env):
    """Base class for all Simple Agar environments.

    This class implements the core logic of Simple Agar. Players are represented by circles, and
    the goal is to eat pellets and other players to grow in size. Player mass decays each turn
    at a rate proportional to its current value. Players receive a reward at each timestep equal
    to the difference in mass between the previous timestep and the current timestep as well as a
    high negative reward for remaining at the base mass. The episode terminates when all players
    but one have been eliminated.

    Parameters
    ----------
    num_players : int
        Number of players in the game.
    num_pellets : int
        Number of pellets in the game.
    pellet_mass : float
        Static mass of each pellet.
    player_mass_base : float
        Base mass of each player.
    player_mass_decay : float
        Decay rate of player mass.
    player_speed_exp : float
        Exponent applied to player mass to determine speed.
    player_speed_scale : float
        Constant scaling factor applied to player speed.
    sqrt_mass_to_radius : float
        Scaling factor applied to the square root of player mass to determine player radius.
    base_mass_penalty : float
        Penalty applied to the reward for remaining at the base mass.

    Attributes
    ----------
    max_player_mass : float
        Maximum player mass.
    action_space : gym.spaces.MultiDiscrete,
        Action space for the environment. Actions are noop, right, up, left, down for each player.
    observation_space : gym.spaces.Dict
        Observation space for the environment. Observations are player masses, player locations,
        and pellet locations.
    """

    def __init__(
        self,
        num_players: int = 1,
        num_pellets: int = 10,
        pellet_mass: float = 1.0,
        player_mass_base: float = 4.0,
        player_mass_decay: float = 0.999,
        player_speed_exp: float = -0.44,
        player_speed_scale: float = 0.08,
        sqrt_mass_to_radius: float = 0.01,
        base_mass_penalty: float = -0.04,
    ):

        self.num_players = num_players
        self.num_pellets = num_pellets
        self.pellet_mass = pellet_mass
        self.pellet_radius = np.sqrt(self.pellet_mass) * sqrt_mass_to_radius
        self.player_mass_base = player_mass_base
        self.player_mass_decay = player_mass_decay
        self.player_speed_exp = player_speed_exp
        self.player_speed_scale = player_speed_scale
        self.sqrt_mass_to_radius = sqrt_mass_to_radius
        self.base_mass_penalty = base_mass_penalty

        self.max_player_mass = (np.sqrt(2) / self.sqrt_mass_to_radius) ** 2

        # actions are noop, right, up, left, down for each player
        self.action_space = MultiDiscrete(np.full(num_players, 5))
        self._action_to_direction = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ],
            dtype=np.float32,
        )

        # players and pellet positions
        self.observation_space = Dict(
            {
                "player_masses": Box(low=0, high=np.inf, shape=(num_players,)),
                "player_is_alive": MultiBinary(num_players),
                "player_locations": Box(low=0, high=1, shape=(num_players, 2)),
                "pellet_locations": Box(low=0, high=1, shape=(num_pellets, 2)),
            }
        )

        # rendering setup
        self._window = None
        self._clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Places players and pellets at random locations. Sets player masses to the base mass.
        """
        super().reset(seed=seed, options=options)

        self._player_masses = np.full(
            self.num_players, self.player_mass_base, dtype=np.float32
        )
        self._player_is_alive = np.full(self.num_players, True, dtype=np.bool)
        self._player_locations = self.np_random.random(
            (self.num_players, 2), dtype=np.float32
        )
        self._pellet_locations = self.np_random.random(
            (self.num_pellets, 2), dtype=np.float32
        )

        self._update_player_distances()
        self._update_player_radii()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, actions: np.ndarray):
        """Advance the environment by one timestep.

        Substeps:
            1. Update player locations using actions.
            2. Update player-to-player and player-to-pellet distances.
            3. Update player radii.
            4. Update player masses based on consumed pellets or players consumed and apply
              decay.
            5. Remove players that have been consumed or have gone off screen.
            6. Respawn consumed pellets.

        Parameters
        ----------
        actions : np.ndarray, shape (num_players,)
            Actions of each player.
        """
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

    def render(self, window_size=WINDOW_SIZE, fps=FPS, highlight_player_index=None):
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((window_size, window_size))

        if self._clock is None:
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        for i in range(self.num_players):
            pygame.draw.circle(
                canvas,
                (0, 255, 0) if highlight_player_index == i else (0, 0, 255),
                (
                    self._player_locations[i][0] * window_size,
                    (1 - self._player_locations[i][1]) * window_size,
                ),
                self._player_radii[i] * window_size,
            )

        for i in range(self.num_pellets):
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (
                    self._pellet_locations[i][0] * window_size,
                    (1 - self._pellet_locations[i][1]) * window_size,
                ),
                self.pellet_radius * window_size,
            )

        self._window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self._clock.tick(fps)

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()

    def _update_player_locations(self, actions):
        # force dead player actions to be noop
        actions[~self._player_is_alive] = 0

        # update player locations
        self._player_locations = (
            self._player_locations
            + self._action_to_direction[actions]
            * np.power(
                np.clip(self._player_masses, a_min=1e-5, a_max=None),
                self.player_speed_exp,
                dtype=np.float32,
            )[:, np.newaxis]
            * self.player_speed_scale
        )

        # find players that moved off screen
        return np.where(
            self._player_is_alive
            & np.any(
                (self._player_locations < 0) | (self._player_locations >= 1), axis=-1
            )
        )

    def _update_player_radii(self):
        self._player_radii = np.sqrt(self._player_masses) * self.sqrt_mass_to_radius

    def _update_player_distances(self):
        self._player_to_player_distances = distance_matrix(
            self._player_locations,
            self._player_locations,
        )
        self._player_to_pellet_distances = distance_matrix(
            self._player_locations,
            self._pellet_locations,
        )

    def _update_player_masses(self):
        # find eaten players
        player_eats_player = (
            (self._player_to_player_distances < self._player_radii[:, np.newaxis])
            & (self._player_radii[:, np.newaxis] > self._player_radii[np.newaxis, :])
        ).astype(np.float32)

        # find eaten pellets
        player_eats_pellet = (
            (
                self._player_to_pellet_distances
                < self._player_radii[:, np.newaxis] + self.pellet_radius
            )
            & (self._player_radii[:, np.newaxis] > self.pellet_radius)
        ).astype(np.float32)

        players_eaten = np.where(np.any(player_eats_player, axis=0))
        pellets_eaten = np.where(np.any(player_eats_pellet, axis=0))

        # divide mass evenly among players that ate the same player or pellet
        player_eats_player[:, players_eaten] /= np.sum(
            player_eats_player[:, players_eaten], axis=0, keepdims=True
        )
        player_eats_pellet[:, pellets_eaten] /= np.sum(
            player_eats_pellet[:, pellets_eaten], axis=0, keepdims=True
        )

        # add mass gained from eating pellets and players
        self._player_masses += np.sum(player_eats_pellet * self.pellet_mass, axis=-1)
        self._player_masses += np.sum(player_eats_player * self._player_masses, axis=-1)

        # decay player mass
        self._player_masses[self._player_is_alive] = np.clip(
            self._player_masses[self._player_is_alive] * self.player_mass_decay,
            a_min=self.player_mass_base,
            a_max=None,
        )

        return players_eaten, pellets_eaten

    def _update_player_is_alive(self, players_eaten, players_off_screen):
        # kill players that are off screen or eaten
        self._player_is_alive[players_eaten] = False
        self._player_is_alive[players_off_screen] = False

        # set dead players to zero mass
        self._player_masses[~self._player_is_alive] = 0

    def _update_pellet_locations(self, pellets_eaten):
        # respawn eaten pellets in random locations
        self._pellet_locations[pellets_eaten] = self.np_random.random(
            self._pellet_locations[pellets_eaten].shape
        )

        # update distances to respawned pellets
        self._player_to_pellet_distances[:, pellets_eaten[0]] = distance_matrix(
            self._player_locations, self._pellet_locations[pellets_eaten]
        )

    def _get_observation(self):
        return {
            "player_masses": self._player_masses / self.max_player_mass,
            "player_is_alive": self._player_is_alive,
            "player_locations": self._player_locations,
            "pellet_locations": self._pellet_locations,
        }

    def _get_reward(self, prev_masses):
        # mass delta is reward
        reward = self._player_masses - prev_masses

        # penalize players that stay at base mass
        reward[self._player_masses == self.player_mass_base] += self.base_mass_penalty

        return reward

    def _get_terminated(self):
        num_players_alive = np.count_nonzero(self._player_is_alive)
        return (
            num_players_alive == 0 if self.num_players == 1 else num_players_alive <= 1
        )

    def _get_truncated(self):
        return False

    def _get_info(self):
        return {
            "player_radii": self._player_radii,
            "player_to_player_distances": self._player_to_player_distances,
            "player_to_pellet_distances": self._player_to_pellet_distances,
        }
