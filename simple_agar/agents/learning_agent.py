import torch
from torch.nn import Module
from typing import SupportsInt, SupportsFloat, List, Tuple

from simple_agar.agents.base_agent import BaseAgent


class LearningAgent(BaseAgent):
    """An agent that learns to play simple agar using a policy-based model."""

    def __init__(self, policy_model: Module):
        super().__init__()
        self.policy_model = policy_model

    def act(self, observation, info):
        log_policy = self.policy_model(observation, info)
        action = torch.multinomial(torch.exp(log_policy), 1)
        log_action_prob = torch.gather(log_policy, 1, action)
        return action.squeeze(-1), log_action_prob.squeeze(-1)
