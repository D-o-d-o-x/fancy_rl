import torch
import gymnasium as gym
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record import VideoRecorder
from abc import ABC

from fancy_rl.loggers import TerminalLogger

class Algo(ABC):
    def __init__(
        self,
        env_spec,
        loggers,
        optimizers,
        learning_rate,
        n_steps,
        batch_size,
        n_epochs,
        gamma,
        total_timesteps,
        eval_interval,
        eval_deterministic,
        entropy_coef,
        critic_coef,
        normalize_advantage,
        device=None,
        eval_episodes=10,
        env_spec_eval=None,
    ):
        self.env_spec = env_spec
        self.env_spec_eval = env_spec_eval if env_spec_eval is not None else env_spec
        self.loggers = loggers if loggers != None else [TerminalLogger(None, None)]
        self.optimizers = optimizers
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.eval_interval = eval_interval
        self.eval_deterministic = eval_deterministic
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.normalize_advantage = normalize_advantage
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_episodes = eval_episodes

    def make_env(self, eval=False):
        """Creates an environment and wraps it if necessary."""
        env_spec = self.env_spec_eval if eval else self.env_spec
        if isinstance(env_spec, str):
            env = gym.make(env_spec)
            env = GymWrapper(env).to(self.device)
        elif callable(env_spec):
            env = env_spec()
            if isinstance(env, gym.Env):
                env = GymWrapper(env).to(self.device)
        elif isinstance(env, gym.Env):
            env = GymWrapper(env).to(self.device)
        else:
            raise ValueError("env_spec must be a string or a callable that returns an environment.")
        return env

    def train_step(self, batch):
        raise NotImplementedError("train_step method must be implemented in subclass.")

    def train(self):
        raise NotImplementedError("train method must be implemented in subclass.")

    def evaluate(self, epoch):
        raise NotImplementedError("evaluate method must be implemented in subclass.")

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()