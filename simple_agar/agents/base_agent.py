import numpy as np


class BaseAgent:
    def act(self, observation, info):
        raise NotImplementedError(
            "act method not implemented for {}".format(self.__class__.__name__)
        )
