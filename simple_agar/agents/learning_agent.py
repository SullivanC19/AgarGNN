import torch
from torch.nn import Module
from typing import SupportsInt, SupportsFloat, List

from simple_agar.agents.base_agent import BaseAgent
from simple_agar.envs.base_world import BaseWorld

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearningAgent(BaseAgent):
    def __init__(self, policy_module: Module):
        super().__init__()
        self.policy_module = policy_module
        self.chosen_action_log_probs = []

    def act(self, observation, info) -> SupportsInt:
        log_policy = self.policy_module(observation, info)
        action = torch.multinomial(torch.exp(log_policy), 1)[0]
        self.chosen_action_log_probs.append(log_policy[action])
        return action
    
    def loss(self, episode_rewards: List[SupportsFloat], discount: SupportsFloat = 1) -> SupportsFloat:
        assert len(episode_rewards) == len(self.chosen_action_log_probs)

        # get discounted returns
        episode_rewards = torch.tensor(episode_rewards, dtype=torch.float).to(device)
        discount = torch.pow(discount, torch.arange(len(episode_rewards))).to(device)
        discounted_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards * discount, dims=(0,)), dim=0), dims=(0,)) / discount
        log_probs = torch.stack(self.chosen_action_log_probs).to(device)

        return -torch.sum(log_probs * discounted_returns)
    
    def reset(self):
        self.chosen_action_log_probs.clear()
