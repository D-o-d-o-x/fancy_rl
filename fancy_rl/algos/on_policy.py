import torch
import gymnasium as gym
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import ExplorationType, set_exploration_type

from fancy_rl.loggers import TerminalLogger
from fancy_rl.algos.algo import Algo

class OnPolicy(Algo):
    def __init__(
        self,
        env_spec,
        optimizers,
        loggers=None,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        total_timesteps=1e6,
        eval_interval=2048,
        eval_deterministic=True,
        entropy_coef=0.01,
        critic_coef=0.5,
        normalize_advantage=True,
        env_spec_eval=None,
        eval_episodes=10,
        device=None,
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(
            env_spec=env_spec,
            loggers=loggers,
            optimizers=optimizers,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            total_timesteps=total_timesteps,
            eval_interval=eval_interval,
            eval_deterministic=eval_deterministic,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=normalize_advantage,
            device=device,
            env_spec_eval=env_spec_eval,
            eval_episodes=eval_episodes,
        )

        # Create collector
        self.collector = SyncDataCollector(
            create_env_fn=lambda: self.make_env(eval=False),
            policy=self.actor,
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

    def pre_process_batch(self, batch):
        return batch

    def post_process_batch(self, batch):
        pass

    def train_step(self, batch):
        batch = self.pre_process_batch(batch)
        
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        losses = self.loss_module(batch)
        loss = sum(losses.values())  # Sum all losses
        loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        self.post_process_batch(batch)
        
        return loss

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
                        logger.log_scalar("loss", loss.item(), step=collected_frames)

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
                    policy=self.actor,
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