import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from typing import List, SupportsFloat

from simple_agar.agents.learning_agent import LearningAgent
from simple_agar.agents.greedy_agent import GreedyAgent
from simple_agar.agents.random_agent import RandomAgent
from simple_agar.agents.base_agent import BaseAgent
from models.mlp_model import MLPModel

from argparse import ArgumentParser

from constants import NUM_EPISODES, MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, HIDDEN_LAYERS, HIDDEN_SIZE, NEGATIVE_SLOPE, DIR_SAVED_MODELS, DIR_RUNS, DIR_RESULTS, K_PELLETS, K_PLAYERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_agent(
        agent: BaseAgent,
        env: gym.Env,
        num_episodes: int,
        render: bool = False
) -> List[SupportsFloat]:
    final_masses = []
    for _ in tqdm(range(num_episodes)):
        observation, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation, info)
            observation, _, terminated, truncated, info = env.step(action)
            if render:
                env.render()

        final_masses.append(observation["player_masses"][env.player_idx] * env.max_player_mass)
        
    env.close()
    return np.mean(final_masses), np.std(final_masses)

def train_model(
        model: torch.nn.Module,
        env: gym.Env,
        num_episodes: int,
        model_save_rate=MODEL_SAVE_RATE,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        f_model=None,
        writer=None
    ) -> List[SupportsFloat]:
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # final_masses = []
    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        episode_rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
        
        optimizer.zero_grad()
        loss, total_discounted_return = agent.loss(episode_rewards, discount_factor)
        loss.backward()
        optimizer.step()
        agent.reset()

        if f_model is not None and (i + 1) % model_save_rate == 0:
            torch.save(model.state_dict(), f_model)

        unnormalized_final_mass = observation["player_masses"][env.player_idx] * env.max_player_mass
        total_episode_reward = sum(episode_rewards)
        if writer is not None:
            writer.add_scalar('Final Mass', unnormalized_final_mass, i)
            writer.add_scalar('Total Discounted Return', total_discounted_return, i)
            writer.add_scalar('Total Reward', total_episode_reward, i)
            writer.add_scalar('Loss', loss, i)
        # final_masses.append(unnormalized_final_mass)
    
    env.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--test", action="store_true")
    mode_group.add_argument("--show", action="store_true")
    parser.add_argument("--env", choices=["pellet", "greedy"], required=True)
    parser.add_argument("--episodes", default=NUM_EPISODES)
    parser.add_argument("--k_pellets", default=K_PELLETS)
    parser.add_argument("--k_players", default=K_PLAYERS)
    parser.add_argument("--hidden_layers", default=HIDDEN_LAYERS)
    parser.add_argument("--hidden_size", default=HIDDEN_SIZE)
    # parser.add_argument("--batch_size")
    args = parser.parse_args()

    env_id = "simple_agar/PelletEatingEnv"
    if (args.env == "greedy"):
        env_id = "simple_agar/GreedyOpponentEnv"
    env = gym.make(env_id)
    
    model_name = f"mlp_model_k_pellets={int(args.k_pellets)}_lr={LEARNING_RATE}"
    model = MLPModel(env, k_pellets=int(args.k_pellets))
    model = model.to(device)
    # model.load_state_dict(torch.load(f_model))
    agent = LearningAgent(model)
    # agent = GreedyAgent()

    if (args.train):
        f_model = os.path.join(DIR_SAVED_MODELS, args.env, model_name)
        writer = SummaryWriter(f_model)
        train_model(model, env, int(args.episodes), MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, f_model, writer)      
    elif (args.test):
        final_masses = run_agent(agent, env, int(args.episodes), render=False)
    else:
        final_masses = run_agent(agent, env, int(args.episodes), render=True)