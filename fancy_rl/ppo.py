import torch
import gymnasium as gym
from fancy_rl.policy import Policy
from fancy_rl.loggers import TerminalLogger
from fancy_rl.on_policy import OnPolicy
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

class PPO(OnPolicy):
    def __init__(
        self,
        policy,
        env_fn,
        loggers=None,
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
        device=None,
        clip_epsilon=0.2,
        **kwargs
    ):
        if loggers is None:
            loggers = [TerminalLogger(push_interval=1)]

        super().__init__(
            policy=policy,
            env_fn=env_fn,
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
            device=device,
            **kwargs
        )
        
        self.clip_epsilon = clip_epsilon
        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.policy,
            average_gae=False,
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.policy,
            clip_epsilon=self.clip_epsilon,
            loss_critic_type='MSELoss',
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            normalize_advantage=self.normalize_advantage,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss_module(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(self):
        self.env = self.env_fn()
        self.env.reset(seed=self.kwargs.get("seed", None))

        state = self.env.reset(seed=self.kwargs.get("seed", None))
        episode_return = 0
        episode_length = 0
        for t in range(self.total_timesteps):
            rollout = self.collect_rollouts(state)
            for batch in self.get_batches(rollout):
                loss = self.train_step(batch)
                for logger in self.loggers:
                    logger.log({
                        "loss": loss.item()
                    }, epoch=t)
                    
                if (t + 1) % self.eval_interval == 0:
                    self.evaluate(t)
