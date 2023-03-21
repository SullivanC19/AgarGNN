import torch
from torch.nn import Module
from typing import SupportsInt, SupportsFloat, List, Tuple

from simple_agar.agents.base_agent import BaseAgent
from simple_agar.envs.base_world import BaseWorld


class LearningAgent(BaseAgent):
    def __init__(self, policy_module: Module):
        super().__init__()
        self.policy_module = policy_module

    def act(self, observation, info):
        log_policy = self.policy_module(observation, info)
        action = torch.multinomial(torch.exp(log_policy), 1)
        log_action_prob = torch.gather(log_policy, 1, action.clone()).clone()
        return action.squeeze(-1), log_action_prob.squeeze(-1)
