

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from typing import List, SupportsFloat

from simple_agar.agents.learning_agent import LearningAgent
from simple_agar.agents.greedy_agent import GreedyAgent
from simple_agar.agents.random_agent import RandomAgent
from simple_agar.agents.base_agent import BaseAgent
from models.mlp_model import MLPModel

from constants import MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, HIDDEN_LAYERS, HIDDEN_SIZE, NEGATIVE_SLOPE, DIR_SAVED_MODELS, DIR_RUNS, DIR_RESULTS, WINDOW_SIZE, FPS

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
    return final_masses

def train_model(
        model: torch.nn.Module,
        env: gym.Env,
        num_episodes: int,
        f_model=None,
        model_save_rate=MODEL_SAVE_RATE,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        writer=None
    ) -> List[SupportsFloat]:
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    final_masses = []
    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        episode_rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
        
        optimizer.zero_grad()
        agent.loss(episode_rewards, discount_factor).backward()
        optimizer.step()
        agent.reset()

        if f_model is not None and (i + 1) % model_save_rate == 0:
            torch.save(model.state_dict(), f_model)

        unnormalized_final_mass = observation["player_masses"][env.player_idx] * env.max_player_mass
        if writer is not None:
            writer.add_scalar('Final Mass', unnormalized_final_mass, i)
        final_masses.append(unnormalized_final_mass)

    env.close()
    return final_masses

if __name__ == '__main__':
    model_name = "mlp_model_k="
    f_run = os.path.join(DIR_RUNS, "pellet_eating", MODEL_NAME)
    f_model = os.path.join(DIR_SAVED_MODELS, "pellet_eating", MODEL_NAME)

    writer = SummaryWriter(f"{DIR_RUNS}/pellet_eating/{MODEL_NAME}")

    env = gym.make(
        'simple_agar/PelletEatingEnv',
        "num_pellets": 10)
    model = MLPModel(env, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE, k_pellets=K_PELLETS, negative_slope=NEGATIVE_SLOPE)
    model = model.to(device)
    model.load_state_dict(torch.load(F_MODEL))
    agent = LearningAgent(model)

    final_masses = train_model(model, env, NUM_EPISODES, F_MODEL, MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, writer)    