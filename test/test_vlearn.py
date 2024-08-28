import pytest
import torch
import numpy as np
from fancy_rl import VLEARN
import gymnasium as gym

@pytest.fixture
def simple_env():
    return gym.make('CartPole-v1')

def test_vlearn_instantiation():
    vlearn = VLEARN("CartPole-v1")
    assert isinstance(vlearn, VLEARN)

@pytest.mark.parametrize("learning_rate", [1e-4, 3e-4, 1e-3])
@pytest.mark.parametrize("n_steps", [1024, 2048])
@pytest.mark.parametrize("batch_size", [32, 64, 128])
@pytest.mark.parametrize("gamma", [0.95, 0.99])
@pytest.mark.parametrize("mean_bound", [0.05, 0.1])
@pytest.mark.parametrize("cov_bound", [0.0005, 0.001])
def test_vlearn_initialization_with_different_hps(learning_rate, n_steps, batch_size, gamma, mean_bound, cov_bound):
    vlearn = VLEARN(
        "CartPole-v1",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        mean_bound=mean_bound,
        cov_bound=cov_bound
    )
    assert vlearn.learning_rate == learning_rate
    assert vlearn.n_steps == n_steps
    assert vlearn.batch_size == batch_size
    assert vlearn.gamma == gamma
    assert vlearn.mean_bound == mean_bound
    assert vlearn.cov_bound == cov_bound

def test_vlearn_predict(simple_env):
    vlearn = VLEARN("CartPole-v1")
    obs, _ = simple_env.reset()
    action, _ = vlearn.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == simple_env.action_space.shape

def test_vlearn_learn():
    vlearn = VLEARN("CartPole-v1", n_steps=64, batch_size=32)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    for _ in range(64):
        action, _ = vlearn.predict(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        vlearn.store_transition(obs, action, reward, done, next_obs)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
    
    loss = vlearn.learn()
    assert isinstance(loss, dict)
    assert "policy_loss" in loss
    assert "value_loss" in loss

def test_vlearn_training(simple_env):
    vlearn = VLEARN("CartPole-v1", total_timesteps=10000)
    
    initial_performance = evaluate_policy(vlearn, simple_env)
    vlearn.train()
    final_performance = evaluate_policy(vlearn, simple_env)
    
    assert final_performance > initial_performance, "VLearn should improve performance after training"

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