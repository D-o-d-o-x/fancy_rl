import torch
from abc import ABC, abstractmethod
from torchrl.record.loggers import Logger
from torch.optim import Adam
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record import VideoRecorder
import gymnasium as gym
try:
    import fancy_gym
except ImportError:
    pass

class OnPolicy(ABC):
    def __init__(
        self,
        policy,
        env_spec,
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
        clip_range=0.2,
        device=None,
        eval_episodes=10,
        env_spec_eval=None,
    ):
        self.policy = policy
        self.env_spec = env_spec
        self.env_spec_eval = env_spec_eval if env_spec_eval is not None else env_spec
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
        self.clip_range = clip_range
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_episodes = eval_episodes

        # Create collector
        self.collector = SyncDataCollector(
            create_env_fn=lambda: self.make_env(eval=False),
            policy=self.policy,
            frames_per_batch=self.n_steps,
            total_frames=self.total_timesteps,
            device=self.device,
            storing_device=self.device,
            max_frames_per_traj=-1,
        )

        # Create data buffer
        self.sampler = SamplerWithoutReplacement()
        self.data_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.n_steps),
            sampler=self.sampler,
            batch_size=self.batch_size,
        )

    def make_env(self, eval=False):
        """Creates an environment and wraps it if necessary."""
        env_spec = self.env_spec_eval if eval else self.env_spec
        if isinstance(env_spec, str):
            env = gym.make(env_spec)
            env = GymWrapper(env)
        elif callable(env_spec):
            env = env_spec()
            if isinstance(env, gym.Env):
                env = GymWrapper(env)
        else:
            raise ValueError("env_spec must be a string or a callable that returns an environment.")
        return env

    def train(self):
        collected_frames = 0

        for t, data in enumerate(self.collector):
            frames_in_batch = data.numel()
            collected_frames += frames_in_batch

            for _ in range(self.n_epochs):
                with torch.no_grad():
                    data = self.adv_module(data)
                data_reshape = data.reshape(-1)
                self.data_buffer.extend(data_reshape)

                for batch in self.data_buffer:
                    batch = batch.to(self.device)
                    loss = self.train_step(batch)
                    for logger in self.loggers:
                        logger.log_scalar({"loss": loss.item()}, step=collected_frames)

            if (t + 1) % self.eval_interval == 0:
                self.evaluate(t)

            self.collector.update_policy_weights_()

    def evaluate(self, epoch):
        eval_env = self.make_env(eval=True)
        eval_env.eval()

        test_rewards = []
        for _ in range(self.eval_episodes):
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                td_test = eval_env.rollout(
                    policy=self.policy,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                    max_steps=10_000_000,
                )
                reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                test_rewards.append(reward.cpu())
                eval_env.apply(dump_video)
        
        avg_return = torch.cat(test_rewards, 0).mean().item()
        for logger in self.loggers:
            logger.log_scalar({"eval_avg_return": avg_return}, step=epoch)

    @abstractmethod
    def train_step(self, batch):
        pass

def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
