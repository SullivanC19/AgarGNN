import numpy as np

from simple_agar.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """An agent that takes random actions."""

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation, info):
        return (
            self.action_space.sample(),
            0.0,
        )  # log action probability is not used for random agent
