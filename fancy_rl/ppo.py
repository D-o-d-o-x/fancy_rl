import torch
import torch.nn as nn
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import get_logger
from on_policy import OnPolicy
from policy import Actor, Critic
import gymnasium as gym

class PPO(OnPolicy):
    def __init__(
        self,
        env_spec,
        loggers=None,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        actor_activation_fn="ReLU",
        critic_activation_fn="ReLU",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        total_timesteps=1e6,
        eval_interval=2048,
        eval_deterministic=True,
        entropy_coef=0.01,
        critic_coef=0.5,
        normalize_advantage=True,
        clip_range=0.2,
        device=None,
        env_spec_eval=None,
        eval_episodes=10,
    ):
        # Initialize environment to get observation and action space sizes
        env = self.make_env(env_spec)
        obs_space = env.observation_space
        act_space = env.action_space

        actor_activation_fn = getattr(nn, actor_activation_fn)
        critic_activation_fn = getattr(nn, critic_activation_fn)

        self.actor = Actor(obs_space, act_space, hidden_sizes=actor_hidden_sizes, activation_fn=actor_activation_fn)
        self.critic = Critic(obs_space, hidden_sizes=critic_hidden_sizes, activation_fn=critic_activation_fn)

        super().__init__(
            policy=self.actor,
            env_spec=env_spec,
            loggers=loggers,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            total_timesteps=total_timesteps,
            eval_interval=eval_interval,
            eval_deterministic=eval_deterministic,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=normalize_advantage,
            clip_range=clip_range,
            device=device,
            env_spec_eval=env_spec_eval,
            eval_episodes=eval_episodes,
        )

        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=False,
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_range,
            loss_critic_type='MSELoss',
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            normalize_advantage=self.normalize_advantage,
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def train_step(self, batch):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss = self.loss_module(batch)
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return loss
