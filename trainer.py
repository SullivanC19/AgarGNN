import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gymnasium as gym

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from typing import List, SupportsFloat

from simple_agar.agents.learning_agent import LearningAgent
from constants import MODEL_SAVE_RATE, DISCOUNT_FACTOR, LEARNING_RATE


def train_model(
    model: torch.nn.Module,
    env: gym.Env,
    num_episodes: int,
    model_save_rate=MODEL_SAVE_RATE,
    discount_factor=DISCOUNT_FACTOR,
    learning_rate=LEARNING_RATE,
    f_model: str = None,
    writer: SummaryWriter = None,
):
    """Train a model in an environment for a given number of episodes.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    env : gym.Env
        The simple agar environment to train the model in.
    num_episodes : int
        The number of episodes to train the model for.
    model_save_rate : int, optional
        The number of episodes between each model save
    discount_factor : float, optional
        The discount factor for the rewards
    learning_rate : float, optional
        The learning rate for the model
    f_model : str, optional
        The file to save the model to. If not provided, the model will not be saved.
    writer : SummaryWriter, optional
        The tensorboard writer to log to.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    agent = LearningAgent(model)

    for i in tqdm(range(num_episodes)):
        observation, info = env.reset()
        episode_rewards = []
        log_action_probs = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, log_action_prob = agent.act(observation, info)

            if (
                isinstance(action, torch.Tensor)
                and action.shape[0] > 1
                and hasattr(env, "player_idx")
            ):
                action = action[env.player_idx]

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

        final_mass = (
            np.mean(observation["player_masses"][player_idx]) * env.max_player_mass
        )
        if writer is not None:
            writer.add_scalar("Final Mass", final_mass, i)
            writer.add_scalar("Loss", loss, i)

    env.close()


def reinforce_loss(
    episode_rewards: List[torch.Tensor],
    log_action_probs: List[torch.Tensor],
    discount: SupportsFloat,
) -> torch.Tensor:
    """Compute the REINFORCE algorithm loss.

    Parameters
    ----------
    episode_rewards : List[torch.Tensor]
        The rewards for each timestep.
    log_action_probs : List[torch.Tensor]
        The log probabilities of the actions taken in each timestep.
    discount : float
        The discount factor for the rewards.

    Returns
    -------
    torch.Tensor
        The REINFORCE loss.
    """

    assert len(episode_rewards) == len(
        log_action_probs
    ), "Rewards and log action probabilities must be the same length."

    episode_rewards = torch.stack(episode_rewards)
    log_action_probs = torch.stack(log_action_probs)
    discount = torch.pow(discount, torch.arange(len(episode_rewards)))[:, None]

    # compute discounted returns
    discounted_returns = (
        torch.flip(
            torch.cumsum(torch.flip(episode_rewards * discount, dims=(0,)), dim=0),
            dims=(0,),
        )
        / discount
    )

    # sum discounted returns over episodes and agents
    return -torch.sum(log_action_probs * discounted_returns)
