

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from simple_agar.agents.learning_agent import LearningAgent
from simple_agar.agents.greedy_agent import GreedyAgent
from models.mlp_model import MLPModel

NUM_EPISODES = 10000000
DISCOUNT_FACTOR = .99
LEARNING_RATE = 1e-3
K_PELLETS = 1
HIDDEN_LAYERS = 3
HIDDEN_SIZE = 32
NEGATIVE_SLOPE = 0.2

MODEL_NAME = f"pellet_eating_mlp_model_h={HIDDEN_LAYERS}_s={HIDDEN_SIZE}_k={K_PELLETS}_lr={LEARNING_RATE}"
F_MODEL = f"saved_models/{MODEL_NAME}.pt"

if __name__ == '__main__':
    writer = SummaryWriter(f"runs/pellet_eating/{MODEL_NAME}")

    config = {
        "num_pellets": 10,
        # "render_mode": "human"
    }

    env = gym.make('simple_agar/PelletEatingEnv', **config)
    model = MLPModel(env, hidden_layers=HIDDEN_LAYERS, hidden_size=HIDDEN_SIZE, k_pellets=K_PELLETS, negative_slope=NEGATIVE_SLOPE)
    # model.load_state_dict(torch.load(F_MODEL))
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

        unnormalized_final_mass = observation["player_masses"][env.player_idx] * env.max_player_mass
        writer.add_scalar('Final Mass', unnormalized_final_mass, i)

    env.close()