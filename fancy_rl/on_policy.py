import torch
from abc import ABC, abstractmethod
from fancy_rl.loggers import Logger
from torch.optim import Adam

class OnPolicy(ABC):
    def __init__(
        self,
        policy,
        env_fn,
        loggers,
        learning_rate,
        n_steps,
        batch_size,
        n_epochs,
        gamma,
        gae_lambda,
        total_timesteps,
        eval_interval,
        eval_deterministic,
        entropy_coef,
        critic_coef,
        normalize_advantage,
        device=None,
        **kwargs
    ):
        self.policy = policy
        self.env_fn = env_fn
        self.loggers = loggers
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.total_timesteps = total_timesteps
        self.eval_interval = eval_interval
        self.eval_deterministic = eval_deterministic
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.normalize_advantage = normalize_advantage
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.kwargs = kwargs
        self.clip_range = 0.2

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

    def evaluate(self, epoch):
        eval_env = self.env_fn()
        eval_env.reset(seed=self.kwargs.get("seed", None))
        returns = []
        for _ in range(self.kwargs.get("eval_episodes", 10)):
            state = eval_env.reset(seed=self.kwargs.get("seed", None))
            done = False
            total_return = 0
            while not done:
                with torch.no_grad():
                    action = (
                        self.policy.act(state, deterministic=self.eval_deterministic)
                        if self.eval_deterministic
                        else self.policy.act(state)
                    )
                state, reward, done, _ = eval_env.step(action)
                total_return += reward
            returns.append(total_return)

        avg_return = sum(returns) / len(returns)
        for logger in self.loggers:
            logger.log({"eval_avg_return": avg_return}, epoch=epoch)
    
    def collect_rollouts(self, state):
        # Collect rollouts logic
        rollouts = []
        for _ in range(self.n_steps):
            action = self.policy.act(state)
            next_state, reward, done, _ = self.env.step(action)
            rollouts.append((state, action, reward, next_state, done))
            state = next_state
            if done:
                state = self.env.reset(seed=self.kwargs.get("seed", None))
        return rollouts

    def get_batches(self, rollouts):
        data = self.prepare_data(rollouts)
        n_batches = len(data) // self.batch_size
        batches = []
        for _ in range(n_batches):
            batch_indices = torch.randint(0, len(data), (self.batch_size,))
            batch = data[batch_indices]
            batches.append(batch)
        return batches

    def prepare_data(self, rollouts):
        obs, actions, rewards, next_obs, dones = zip(*rollouts)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        data = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones
        }
        data = self.adv_module(data)
        return data

    @abstractmethod
    def train_step(self, batch):
        pass
