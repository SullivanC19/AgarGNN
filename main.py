import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from simple_agar.agents.learning_agent import LearningAgent
from simple_agar.agents.greedy_agent import GreedyAgent
from simple_agar.agents.random_agent import RandomAgent
from simple_agar.agents.base_agent import BaseAgent
from models.mlp_model import MLPModel

from trainer import train_model

from argparse import ArgumentParser

from constants import NUM_EPISODES, MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, DIR_SAVED_MODELS, DIR_RUNS, DIR_RESULTS, K_PELLET, K_PLAYER


def run_agent(
        agent: BaseAgent,
        env: gym.Env,
        num_episodes: int = 1,
        render: bool = False
):
    final_masses = []
    for _ in tqdm(range(num_episodes)):
        observation, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = agent.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            if render:
                env.render()

        final_masses.append(observation["player_masses"][env.player_idx] * env.max_player_mass)
        
    env.close()

    print(f"Average final mass: {np.mean(final_masses)}")
    print(f"Std of final mass: {np.std(final_masses)}")


if __name__ == '__main__':
    parser = ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--test", action="store_true")
    mode_group.add_argument("--show", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--env", choices=["pellet", "greedy", "self"], required=True)
    parser.add_argument("--agent", choices=["learning", "greedy", "random"], default="learning")
    parser.add_argument("--model", choices=["mlp", "gnn"])
    parser.add_argument("--episodes", default=NUM_EPISODES)
    parser.add_argument("--k_pellet", default=K_PELLET)
    parser.add_argument("--k_player", default=K_PLAYER)
    args = parser.parse_args()

    env_id = {
        "pellet": "simple_agar/PelletEatingEnv",
        "greedy": "simple_agar/GreedyOpponentEnv",
    }[args.env]
    env = gym.make(env_id)

    agent = None
    if args.agent == "learning":
        assert args.model is not None, "Model architecture must be specified for learning agent"

        if args.model == "mlp":
            model_name = "mlp_model_k_pellet={}_k_player={}".format(args.k_pellet, args.k_player)
            model = MLPModel(env, k_pellet=int(args.k_pellet), k_player=int(args.k_player))
        elif args.model == "gnn":
            model_name = "gnn_model"
            model = GNNModel(env, k_pellet=int(args.k_pellet), k_player=int(args.k_player))

        d_model = os.path.join(DIR_SAVED_MODELS, args.env)
        f_model = os.path.join(d_model, f"{model_name}.pt")

        if not os.path.exists(d_model):
            os.makedirs(d_model)
        
        if args.load:
            assert os.path.exists(f_model), "Model file does not exist"
            model.load_state_dict(torch.load(f_model))

        agent = LearningAgent(model)

    elif args.agent == "greedy":
        agent = GreedyAgent()

    elif args.agent == "random":
        agent = RandomAgent(env.action_space)


    if args.train:
        assert args.agent == "learning", "Only learning agent can be trained"
        d_run = os.path.join(DIR_RUNS, args.env, model_name)
        writer = SummaryWriter(d_run)
        train_model(model, env, int(args.episodes), MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, f_model, writer)
        
    elif args.test:
        run_agent(agent, env, int(args.episodes))

    elif args.show:
        run_agent(agent, env, render=True)