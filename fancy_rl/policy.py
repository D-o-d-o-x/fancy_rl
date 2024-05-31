import torch
from torch import nn
from torch.distributions import Categorical, Normal
import gymnasium as gym

class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=[64, 64], activation_fn=nn.ReLU):
        super().__init__()
        self.continuous = isinstance(action_space, gym.spaces.Box)
        input_dim = observation_space.shape[-1]
        if self.continuous:
            output_dim = action_space.shape[-1]
        else:
            output_dim = action_space.n
        
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(activation_fn())
            last_dim = size
        
        if self.continuous:
            self.mu_layer = nn.Linear(last_dim, output_dim)
            self.log_std_layer = nn.Linear(last_dim, output_dim)
        else:
            layers.append(nn.Linear(last_dim, output_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.continuous:
            mu = self.mu_layer(x)
            log_std = self.log_std_layer(x)
            return mu, log_std.exp()
        else:
            return self.model(x)

    def act(self, observation, deterministic=False):
        with torch.no_grad():
            if self.continuous:
                mu, std = self.forward(observation)
                if deterministic:
                    action = mu
                else:
                    action_dist = Normal(mu, std)
                    action = action_dist.sample()
            else:
                logits = self.forward(observation)
                if deterministic:
                    action = logits.argmax(dim=-1)
                else:
                    action_dist = Categorical(logits=logits)
                    action = action_dist.sample()
        return action

class Critic(nn.Module):
    def __init__(self, observation_space, hidden_sizes=[64, 64], activation_fn=nn.ReLU):
        super().__init__()
        input_dim = observation_space.shape[-1]
        
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(activation_fn())
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)
