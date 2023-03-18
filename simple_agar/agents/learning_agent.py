import torch
from torch.nn import Module
from typing import SupportsInt, SupportsFloat, List

from simple_agar.agents.base_agent import BaseAgent
from simple_agar.envs.base_world import BaseWorld

class LearningAgent(BaseAgent):
    def __init__(self, policy_module: 'PolicyModule'):
        super().__init__()
        self.policy_module = policy_module
        self.chosen_action_log_probs = []

    def act(self, observation, info) -> SupportsInt:
        log_policy = self.policy_module(observation, info)
        if torch.any(torch.isnan(log_policy)) or torch.any(torch.exp(log_policy) < 0):
            print("WARNING: log_policy has negative values or NaNs")
            print(log_policy)

        action = torch.multinomial(torch.exp(log_policy), 1)[0]
        self.chosen_action_log_probs.append(log_policy[action])
        return action
    
    def loss(self, episode_rewards: List[SupportsFloat], discount=1) -> SupportsFloat:
        assert len(episode_rewards) == len(self.chosen_action_log_probs)

        # get discounted returns
        episode_rewards = torch.tensor(episode_rewards, dtype=torch.float)
        discount = torch.pow(discount, torch.arange(len(episode_rewards)))
        discounted_returns = torch.flip(torch.cumsum(torch.flip(episode_rewards * discount, dims=(0,)), dim=0), dims=(0,)) / discount
        
        return -torch.sum(torch.stack(self.chosen_action_log_probs) * discounted_returns)
    
    def reset(self):
        self.chosen_action_log_probs.clear()


    

