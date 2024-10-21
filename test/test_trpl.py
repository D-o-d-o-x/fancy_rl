import pytest
import numpy as np
from fancy_rl import TRPL
import gymnasium as gym

def simple_env():
    return gym.make('LunarLander-v2', continuous=True)

def test_trpl_instantiation():
    trpl = TRPL(simple_env)
    assert isinstance(trpl, TRPL)

def test_trpl_instantiation_from_str():
    trpl = TRPL('MountainCarContinuous-v0')
    assert isinstance(trpl, TRPL)

@pytest.mark.parametrize("learning_rate", [1e-4, 3e-4, 1e-3])
@pytest.mark.parametrize("n_steps", [1024, 2048])
@pytest.mark.parametrize("batch_size", [32, 64, 128])
@pytest.mark.parametrize("gamma", [0.95, 0.99])
@pytest.mark.parametrize("trust_region_bound_mean", [0.05, 0.1])
@pytest.mark.parametrize("trust_region_bound_cov", [0.0005, 0.001])
def test_trpl_initialization_with_different_hps(learning_rate, n_steps, batch_size, gamma, trust_region_bound_mean, trust_region_bound_cov):
    trpl = TRPL(
        simple_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        trust_region_bound_mean=trust_region_bound_mean,
        trust_region_bound_cov=trust_region_bound_cov
    )
    assert trpl.learning_rate == learning_rate
    assert trpl.n_steps == n_steps
    assert trpl.batch_size == batch_size
    assert trpl.gamma == gamma
    assert trpl.projection.trust_region_bound_mean == trust_region_bound_mean
    assert trpl.projection.trust_region_bound_cov == trust_region_bound_cov

def test_trpl_predict():
    trpl = TRPL(simple_env)
    env = trpl.make_env()
    obs, _ = env.reset()
    action, _ = trpl.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape

def test_trpl_learn():
    trpl = TRPL(simple_env, n_steps=64, batch_size=32)
    env = trpl.make_env()
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

def test_trpl_training():
    trpl = TRPL(simple_env, total_timesteps=10000)
    env = trpl.make_env()
    
    initial_performance = evaluate_policy(trpl, env)
    trpl.train()
    final_performance = evaluate_policy(trpl, env)
    
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