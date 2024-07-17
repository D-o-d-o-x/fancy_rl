import torch
from torch import nn
from typing import Dict, Any, Optional
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from fancy_rl.utils import get_env, get_actor, get_critic
from fancy_rl.modules.vlearn_loss import VLEARNLoss
from fancy_rl.modules.projection import get_vlearn_projection
from fancy_rl.modules.squashed_normal import get_squashed_normal

class VLEARN:
    def __init__(self, env_id: str, device: str = "cpu", **kwargs: Any):
        self.device = torch.device(device)
        self.env = get_env(env_id)
        
        self.projection = get_vlearn_projection(**kwargs.get("projection", {}))
        
        actor = get_actor(self.env, **kwargs.get("actor", {}))
        self.actor = ProbabilisticActor(
            actor,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
            distribution_class=get_squashed_normal(),
            return_log_prob=True
        ).to(self.device)
        self.old_actor = self.actor.clone()
        
        self.critic = ValueOperator(
            module=get_critic(self.env, **kwargs.get("critic", {})),
            in_keys=["observation"]
        ).to(self.device)
        
        self.collector = SyncDataCollector(
            self.env,
            self.actor,
            frames_per_batch=kwargs.get("frames_per_batch", 1000),
            total_frames=kwargs.get("total_frames", -1),
            device=self.device,
        )
        
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(kwargs.get("buffer_size", 100000)),
            batch_size=kwargs.get("batch_size", 256),
        )
        
        self.loss_module = VLEARNLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            old_actor_network=self.old_actor,
            projection=self.projection,
            **kwargs.get("loss", {})
        )
        
        self.optimizers = nn.ModuleDict({
            "policy": torch.optim.Adam(self.actor.parameters(), lr=kwargs.get("lr_policy", 3e-4)),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=kwargs.get("lr_critic", 3e-4))
        })

        self.update_policy_interval = kwargs.get("update_policy_interval", 1)
        self.update_critic_interval = kwargs.get("update_critic_interval", 1)
        self.target_update_interval = kwargs.get("target_update_interval", 1)
        self.polyak_weight_critic = kwargs.get("polyak_weight_critic", 0.995)

    def train(self, num_iterations: int = 1000) -> None:
        for i in range(num_iterations):
            data = next(self.collector)
            self.replay_buffer.extend(data)
            
            batch = self.replay_buffer.sample().to(self.device)
            loss_dict = self.loss_module(batch)
            
            if i % self.update_policy_interval == 0:
                self.optimizers["policy"].zero_grad()
                loss_dict["policy_loss"].backward()
                self.optimizers["policy"].step()
            
            if i % self.update_critic_interval == 0:
                self.optimizers["critic"].zero_grad()
                loss_dict["critic_loss"].backward()
                self.optimizers["critic"].step()
            
            if i % self.target_update_interval == 0:
                self.critic.update_target_params(self.polyak_weight_critic)
            
            self.old_actor.load_state_dict(self.actor.state_dict())
            self.collector.update_policy_weights_()

            if i % 100 == 0:
                eval_reward = self.eval()
                print(f"Iteration {i}, Eval reward: {eval_reward}")

    def eval(self, num_episodes: int = 10) -> float:
        total_reward = 0
        for _ in range(num_episodes):
            td = self.env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = self.actor(td.to(self.device))["action"]
                td = self.env.step(action)
                total_reward += td["reward"].item()
                done = td["done"].item()
        return total_reward / num_episodes

    def save_policy(self, path: str) -> None:
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_policy(self, path: str) -> None:
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))
        self.old_actor.load_state_dict(self.actor.state_dict())