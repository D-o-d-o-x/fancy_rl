import pytest
import numpy as np
from fancy_rl import TRPL
import gymnasium as gym

@pytest.fixture
def simple_env():
    return gym.make('CartPole-v1')

def test_trpl_instantiation():
    trpl = TRPL("CartPole-v1")
    assert isinstance(trpl, TRPL)

@pytest.mark.parametrize("learning_rate", [1e-4, 3e-4, 1e-3])
@pytest.mark.parametrize("n_steps", [1024, 2048])
@pytest.mark.parametrize("batch_size", [32, 64, 128])
@pytest.mark.parametrize("gamma", [0.95, 0.99])
@pytest.mark.parametrize("max_kl", [0.01, 0.05])
def test_trpl_initialization_with_different_hps(learning_rate, n_steps, batch_size, gamma, max_kl):
    trpl = TRPL(
        "CartPole-v1",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        max_kl=max_kl
    )
    assert trpl.learning_rate == learning_rate
    assert trpl.n_steps == n_steps
    assert trpl.batch_size == batch_size
    assert trpl.gamma == gamma
    assert trpl.max_kl == max_kl

def test_trpl_predict(simple_env):
    trpl = TRPL("CartPole-v1")
    obs, _ = simple_env.reset()
    action, _ = trpl.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == simple_env.action_space.shape

def test_trpl_learn():
    trpl = TRPL("CartPole-v1", n_steps=64, batch_size=32)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    for _ in range(64):
        action, _ = trpl.predict(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        trpl.store_transition(obs, action, reward, done, next_obs)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
    
    loss = trpl.learn()
    assert isinstance(loss, dict)
    assert "policy_loss" in loss
    assert "value_loss" in loss

def test_trpl_training(simple_env):
    trpl = TRPL("CartPole-v1", total_timesteps=10000)
    
    initial_performance = evaluate_policy(trpl, simple_env)
    trpl.train()
    final_performance = evaluate_policy(trpl, simple_env)
    
    assert final_performance > initial_performance, "TRPL should improve performance after training"

def evaluate_policy(policy, env, n_eval_episodes=10):
    total_reward = 0
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / n_eval_episodes