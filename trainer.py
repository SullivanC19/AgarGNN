import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from typing import List, SupportsFloat

from simple_agar.agents.learning_agent import LearningAgent
from constants import MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
        model: torch.nn.Module,
        env: gym.Env,
        num_episodes: int,
        model_save_rate=MODEL_SAVE_RATE,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LEARNING_RATE,
        f_model: str = None,
        writer: SummaryWriter = None
    ):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    agent = LearningAgent(model)

    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        episode_rewards = []
        log_action_probs = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, log_action_prob = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(torch.from_numpy(reward))
            log_action_probs.append(log_action_prob)
        
        optimizer.zero_grad()
        loss = reinforce_loss(episode_rewards, log_action_probs, discount_factor)
        loss.backward()
        optimizer.step()

        if f_model is not None and (i + 1) % model_save_rate == 0:
            torch.save(model.state_dict(), f_model)

        player_idx = env.player_idx if hasattr(env, "player_idx") else 0

        final_mass = np.mean(observation["player_masses"][player_idx]) * env.max_player_mass
        if writer is not None:
            writer.add_scalar('Final Mass', final_mass, i)
            writer.add_scalar('Loss', loss, i)

    env.close() 

def reinforce_loss(
        episode_rewards: List[torch.Tensor],
        log_action_probs: List[torch.Tensor],
        discount: SupportsFloat) -> torch.Tensor:
    
    assert len(episode_rewards) == len(log_action_probs)

    episode_rewards = torch.stack(episode_rewards).to(device)
    log_action_probs = torch.stack(log_action_probs).to(device)
    discount = torch.pow(discount, torch.arange(len(episode_rewards)))[:, None].to(device)

    # compute discounted returns
    discounted_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards * discount, dims=(0,)), dim=0), dims=(0,)) / discount

    # sum discounted returns over episodes and agents
    return -torch.sum(log_action_probs * discounted_returns)