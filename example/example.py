import yaml
import torch
from fancy_rl.ppo import PPO
from fancy_rl.policy import Policy
from fancy_rl.loggers import TerminalLogger, WandbLogger
import gymnasium as gym

def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    env_fn = lambda: gym.make("CartPole-v1")
    env = env_fn()

    policy_config = config['policy']
    policy = Policy(env=env, hidden_sizes=policy_config['hidden_sizes'])

    ppo_config = config['ppo']
    loggers_config = config['loggers']

    loggers = []
    for logger_config in loggers_config:
        logger_type = logger_config.pop('type')
        if logger_type == 'terminal':
            loggers.append(TerminalLogger(**logger_config))
        elif logger_type == 'wandb':
            loggers.append(WandbLogger(**logger_config))

    ppo = PPO(policy=policy,
              env_fn=env_fn,
              loggers=loggers,
              **ppo_config)

    ppo.train()

if __name__ == "__main__":
    main("example/config.yaml")
