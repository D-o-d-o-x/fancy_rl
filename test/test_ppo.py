import pytest
import torch
from fancy_rl.ppo import PPO
from fancy_rl.policy import Policy
from fancy_rl.loggers import TerminalLogger
from fancy_rl.utils import make_env

@pytest.fixture
def policy():
    return Policy(input_dim=4, output_dim=2, hidden_sizes=[64, 64])

@pytest.fixture
def loggers():
    return [TerminalLogger()]

@pytest.fixture
def env_fn():
    return make_env("CartPole-v1")

def test_ppo_train(policy, loggers, env_fn):
    ppo = PPO(policy=policy,
              env_fn=env_fn,
              loggers=loggers,
              learning_rate=3e-4,
              n_steps=2048,
              batch_size=64,
              n_epochs=10,
              gamma=0.99,
              gae_lambda=0.95,
              clip_range=0.2,
              total_timesteps=10000,
              eval_interval=2048,
              eval_deterministic=True,
              eval_episodes=5,
              seed=42)
    ppo.train()

def test_ppo_evaluate(policy, loggers, env_fn):
    ppo = PPO(policy=policy,
              env_fn=env_fn,
              loggers=loggers,
              learning_rate=3e-4,
              n_steps=2048,
              batch_size=64,
              n_epochs=10,
              gamma=0.99,
              gae_lambda=0.95,
              clip_range=0.2,
              total_timesteps=10000,
              eval_interval=2048,
              eval_deterministic=True,
              eval_episodes=5,
              seed=42)
    ppo.evaluate(epoch=0)
