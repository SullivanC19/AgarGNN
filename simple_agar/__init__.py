from gymnasium.envs.registration import register


register(
    id="simple_agar/PelletEatingEnv",
    entry_point="simple_agar.envs.entry_points:pellet_eating_env_builder",
    max_episode_steps=1000,
)

register(
    id="simple_agar/GreedyOpponentEnv",
    entry_point="simple_agar.envs.entry_points:greedy_opponent_env_builder",
    max_episode_steps=10000,
)

register(
    id="simple_agar/MultiAgentSelfLearningEnv",
    entry_point="simple_agar.envs.entry_points:multi_agent_self_learning_env_builder",
    max_episode_steps=10000,
)
