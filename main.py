# TODO @sullivanc19 find another fix for this
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gymnasium as gym
import numpy as np

import simple_agar


config = {
    "num_players": 2,
    "num_pellets": 1,}
    
env = gym.make("simple_agar/BaseWorld-v0", **config)
print(env.reset())

for i in range(10):
    print(env.step(np.array([1, 1])))