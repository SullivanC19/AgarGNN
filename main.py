

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from simple_agar.agents.learning_agent import LearningAgent
from models.mlp_model import MLPModel

NUM_EPISODES = 5000
DISCOUNT_FACTOR = .99
LEARNING_RATE = 3e-4
K_PELLETS = 1
HIDDEN_LAYERS = 3
HIDDEN_SIZE = 32

MODEL_NAME = f"pellet_eating_mlp_model_h={HIDDEN_LAYERS}_s={HIDDEN_SIZE}_k={K_PELLETS}"
F_MODEL = f"saved_models/{MODEL_NAME}.pt"

if __name__ == '__main__':
    writer = SummaryWriter('runs/pellet_eating/')

    config = {
        "num_pellets": 1000,
        # "render_mode": "human"
    }

    env = gym.make('simple_agar/PelletEatingEnv', **config)
    model = MLPModel(env, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE, k_pellets=K_PELLETS)
    agent = LearningAgent(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i in tqdm(range(NUM_EPISODES)):
        observation, info = env.reset()
        episode_rewards = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
        
        optimizer.zero_grad()
        agent.loss(episode_rewards, DISCOUNT_FACTOR).backward()
        optimizer.step()
        agent.reset()

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), F_MODEL)

        final_mass = observation["player_masses"][env.player_idx]
        writer.add_scalar('Final Mass', final_mass, i)

    env.close()