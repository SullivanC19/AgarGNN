from gymnasium.envs.registration import register

register(
    id="simple_agar/BaseWorld-v0",
    entry_point="simple_agar.envs.base_world:BaseWorld",
    max_episode_steps=1000,)
