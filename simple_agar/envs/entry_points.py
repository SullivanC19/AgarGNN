from simple_agar.envs.base_world import BaseWorld
from simple_agar.wrappers.single_player import SinglePlayer


def pellet_eating_env_builder(**config):
    """Build an environment where the agent is the only player in the world."""
    config["num_players"] = 1
    base_world = BaseWorld(**config)
    return SinglePlayer(base_world, player_idx=0)


def greedy_opponent_env_builder(**config):
    """Build an environment where the agent competes against three other greedy
    opponents.
    """
    config["num_players"] = 4
    base_world = BaseWorld(**config)
    return SinglePlayer(base_world, player_idx=0)


def multi_agent_self_learning_env_builder(**config):
    """Build an environment where the agent determines the actions of all players
    in the world and competes against itself."""
    config["num_players"] = 4
    base_world = BaseWorld(**config)
    return base_world
