

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
    return final_masses

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
        final_masses.append(unnormalized_final_mass)

    env.close()
    return final_masses

if __name__ == '__main__':
    model_name = "mlp_model_k_pellets=1"

    f_run = os.path.join(DIR_RUNS, "pellet_eating", model_name)
    f_model = os.path.join(DIR_SAVED_MODELS, "pellet_eating", model_name)

    writer = SummaryWriter(f"{DIR_RUNS}/pellet_eating/{model_name}")

    env = gym.make(
        'simple_agar/PelletEatingEnv')
    model = MLPModel(env, k_pellets=1)
    model = model.to(device)
    # model.load_state_dict(torch.load(f_model))
    agent = LearningAgent(model)

    # final_masses = train_model(model, env, NUM_EPISODES, MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, None, None)
    final_masses = train_model(model, env, NUM_EPISODES, MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE, f_model, writer)      
    # final_masses = run_agent(agent, env, NUM_EPISODES, True)