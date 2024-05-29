import gymnasium as gym

def make_env(env_name):
    return lambda: gym.make(env_name)
