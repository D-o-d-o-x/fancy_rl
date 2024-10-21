import pytest
import numpy as np
from fancy_rl import PPO
import gymnasium as gym

def simple_env():
    return gym.make('LunarLander-v2', continuous=True)

def test_ppo_instantiation():
    ppo = PPO(simple_env)
    assert isinstance(ppo, PPO)

def test_ppo_instantiation_from_str():
    ppo = PPO('CartPole-v1')
    assert isinstance(ppo, PPO)

@pytest.mark.parametrize("learning_rate", [1e-4, 3e-4, 1e-3])
@pytest.mark.parametrize("n_steps", [1024, 2048])
@pytest.mark.parametrize("batch_size", [32, 64, 128])
@pytest.mark.parametrize("n_epochs", [5, 10])
@pytest.mark.parametrize("gamma", [0.95, 0.99])
@pytest.mark.parametrize("clip_range", [0.1, 0.2, 0.3])
def test_ppo_initialization_with_different_hps(learning_rate, n_steps, batch_size, n_epochs, gamma, clip_range):
    ppo = PPO(
        simple_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range
    )
    assert ppo.learning_rate == learning_rate
    assert ppo.n_steps == n_steps
    assert ppo.batch_size == batch_size
    assert ppo.n_epochs == n_epochs
    assert ppo.gamma == gamma
    assert ppo.clip_range == clip_range

def test_ppo_predict():
    ppo = PPO(simple_env)
    env = ppo.make_env()
    obs, _ = env.reset()
    action, _ = ppo.predict(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == env.action_space.shape

def test_ppo_learn():
    ppo = PPO(simple_env, n_steps=64, batch_size=32)
    env = ppo.make_env()
    obs, _ = env.reset()
    for _ in range(64):
        action, _ = ppo.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

def test_ppo_training():
    ppo = PPO(simple_env, total_timesteps=10000)
    env = ppo.make_env()
    
    initial_performance = evaluate_policy(ppo, env)
    ppo.train()
    final_performance = evaluate_policy(ppo, env)
    
    assert final_performance > initial_performance, "PPO should improve performance after training"

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