import yaml
import torch
from ppo import PPO
from torchrl.record.loggers import get_logger
import gymnasium as gym

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    env_spec = "CartPole-v1"

    ppo_config = config['ppo']
    actor_config = config['actor']
    critic_config = config['critic']
    loggers_config = config.get('loggers', [])

    loggers = [get_logger(**logger_config) for logger_config in loggers_config]

    ppo = PPO(
        env_spec=env_spec,
        loggers=loggers,
        actor_hidden_sizes=actor_config['hidden_sizes'],
        critic_hidden_sizes=critic_config['hidden_sizes'],
        actor_activation_fn=actor_config['activation_fn'],
        critic_activation_fn=critic_config['activation_fn'],
        **ppo_config
    )

    ppo.train()

if __name__ == "__main__":
    main("example/config.yaml")
