import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
from fancy_rl.utils import is_discrete_space, get_space_shape
from tensordict import TensorDict

class Actor(TensorDictModule):
    def __init__(self, obs_space, act_space, hidden_sizes, activation_fn, device, full_covariance=False):
        self.discrete = is_discrete_space(act_space)
        act_space_shape = get_space_shape(act_space)
        
        if self.discrete and full_covariance:
            raise ValueError("Full covariance is not applicable for discrete action spaces.")
        
        self.full_covariance = full_covariance

        if self.discrete:
            out_features = act_space_shape[-1]
            out_keys = ["action_logits"]
        else:
            if full_covariance:
                out_features = act_space_shape[-1] + (act_space_shape[-1] * (act_space_shape[-1] + 1)) // 2
                out_keys = ["loc", "scale_tril"]
            else:
                out_features = act_space_shape[-1] * 2
                out_keys = ["loc", "scale"]

        actor_module = MLP(
            in_features=get_space_shape(obs_space)[-1],
            out_features=out_features,
            num_cells=hidden_sizes,
            activation_class=getattr(nn, activation_fn),
            device=device
        ).to(device)

        if not self.discrete:
            if full_covariance:
                param_extractor = FullCovarianceNormalParamExtractor(act_space_shape[-1])
            else:
                param_extractor = NormalParamExtractor()
            actor_module = nn.Sequential(actor_module, param_extractor)

        super().__init__(
            module=actor_module,
            in_keys=["observation"],
            out_keys=out_keys
        )

class FullCovarianceNormalParamExtractor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, x):
        loc = x[:, :self.action_dim]
        scale_tril = torch.zeros(x.shape[0], self.action_dim, self.action_dim, device=x.device)
        tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        scale_tril[:, tril_indices[0], tril_indices[1]] = x[:, self.action_dim:]
        scale_tril.diagonal(dim1=-2, dim2=-1).exp_()
        return TensorDict({"loc": loc, "scale_tril": scale_tril}, batch_size=x.shape[0])

class Critic(TensorDictModule):
    def __init__(self, obs_space, hidden_sizes, activation_fn, device):
        critic_module = MLP(
            in_features=get_space_shape(obs_space)[-1],
            out_features=1,
            num_cells=hidden_sizes,
            activation_class=getattr(nn, activation_fn),
            device=device
        ).to(device)
        super().__init__(
            module=critic_module,
            in_keys=["observation"],
            out_keys=["state_value"],
        )