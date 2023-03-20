from simple_agar.envs.base_world import BaseWorld
from simple_agar.wrappers.single_player import SinglePlayer

def pellet_eating_env_builder(**config):
    config["num_players"] = 1
    config["num_pellets"] = 50
    base_world = BaseWorld(**config)
    return SinglePlayer(base_world, player_idx=0)

def greedy_opponent_env_builder(**config):
    config["num_players"] = 10
    base_world = BaseWorld(**config)
    return SinglePlayer(base_world, player_idx=0)

def multi_agent_self_learning_env_builder(**config):
    base_world = BaseWorld(**config)
    return base_world