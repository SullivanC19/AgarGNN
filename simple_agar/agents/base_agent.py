from typing import SupportsFloat, SupportsInt, Tuple, Union

import torch


class BaseAgent:
    """Base class for simple agents.

    Agents are objects that take in the observation and info from a simple
    agar environment and return a sampled action as well as the log probability
    the action was chosen. A single agent may control multiple players in the
    environment by outputting an action and log probability for each player at
    each timestep (e.g. a learning agent controlling multiple players in a
    multi-agent environment).
    """

    def act(
        self, observation, info
    ) -> Tuple[Union[SupportsInt, torch.Tensor], Union[SupportsFloat, torch.Tensor]]:
        """Sample an action from the agent's policy.

        Accept an observation and info from a simple agar environment and return
        a sampled action as well as the log probability that action was chosen.

        Parameters
        ----------
        observation : dict
            The observation from the environment.
        info : dict
            The info from the environment.

        Returns
        -------
        action : Union[SupportsInt, torch.Tensor]
            The sampled action.
        log_action_prob : Union[SupportsFloat, torch.Tensor]
            The log probability that the action was chosen.
        """

        raise NotImplementedError(
            "act method not implemented for {}".format(self.__class__.__name__)
        )
