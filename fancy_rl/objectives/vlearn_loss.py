import torch
from torchrl.objectives import LossModule
from torch.distributions import Normal

class VLEARNLoss(LossModule):
    def __init__(
        self,
        actor_network,
        critic_network,
        old_actor_network,
        gamma=0.99,
        lmbda=0.95,
        entropy_coef=0.01,
        critic_coef=0.5,
        normalize_advantage=True,
        eps=1e-8,
        delta=0.1
    ):
        super().__init__()
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.old_actor_network = old_actor_network
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.normalize_advantage = normalize_advantage
        self.eps = eps
        self.delta = delta

    def forward(self, tensordict):
        # Compute returns and advantages
        with torch.no_grad():
            returns = self.compute_returns(tensordict)
            values = self.critic_network(tensordict)["state_value"]
            advantages = returns - values
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute actor loss
        new_td = self.actor_network(tensordict)
        old_td = self.old_actor_network(tensordict)

        new_dist = Normal(new_td["loc"], new_td["scale"])
        old_dist = Normal(old_td["loc"], old_td["scale"])

        new_log_prob = new_dist.log_prob(tensordict["action"]).sum(-1)
        old_log_prob = old_dist.log_prob(tensordict["action"]).sum(-1)

        ratio = torch.exp(new_log_prob - old_log_prob)
        
        # Compute projection
        kl = torch.distributions.kl.kl_divergence(new_dist, old_dist).sum(-1)
        alpha = torch.where(kl > self.delta, 
                            torch.sqrt(self.delta / (kl + self.eps)), 
                            torch.ones_like(kl))
        proj_loc = alpha.unsqueeze(-1) * new_td["loc"] + (1 - alpha.unsqueeze(-1)) * old_td["loc"]
        proj_scale = torch.sqrt(alpha.unsqueeze(-1)**2 * new_td["scale"]**2 + (1 - alpha.unsqueeze(-1))**2 * old_td["scale"]**2)
        proj_dist = Normal(proj_loc, proj_scale)

        proj_log_prob = proj_dist.log_prob(tensordict["action"]).sum(-1)
        proj_ratio = torch.exp(proj_log_prob - old_log_prob)
        
        policy_loss = -torch.min(
            ratio * advantages,
            proj_ratio * advantages
        ).mean()

        # Compute critic loss
        value_pred = self.critic_network(tensordict)["state_value"]
        critic_loss = 0.5 * (returns - value_pred).pow(2).mean()

        # Compute entropy loss
        entropy_loss = -self.entropy_coef * new_dist.entropy().mean()

        # Combine losses
        loss = policy_loss + self.critic_coef * critic_loss + entropy_loss

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "entropy_loss": entropy_loss,
        }

    def compute_returns(self, tensordict):
        rewards = tensordict["reward"]
        dones = tensordict["done"]
        values = self.critic_network(tensordict)["state_value"]
        
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lmbda * (1 - dones[t]) * last_gae_lam
        
        returns = advantages + values
        
        return returns