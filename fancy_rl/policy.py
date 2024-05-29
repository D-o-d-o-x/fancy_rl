import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def act(self, observation, deterministic=False):
        with torch.no_grad():
            logits = self.forward(observation)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample()
        return action
