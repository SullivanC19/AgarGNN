import gymnasium as gym
from torch.optim import Adam

import simple_agar

from simple_agar.agents.learning_agent import LearningAgent
from models.mlp_model import MLPModel

NUM_EPISODES = 100
DISCOUNT_FACTOR = .999
LEARNING_RATE = 1e-4

if __name__ == '__main__':
    config = {
        "num_pellets": 1000,
    }

    env = gym.make('simple_agar/PelletEatingEnv', config)
    model = MLPModel(env)
    agent = LearningAgent(model)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for i in range(NUM_EPISODES):
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

        print(f"Reward: {reward}")

    env.close()