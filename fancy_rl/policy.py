import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
from fancy_rl.utils import is_discrete_space, get_space_shape

class SharedModule(TensorDictModule):
    def __init__(self, obs_space, hidden_sizes, activation_fn, device):
        if hidden_sizes:
            shared_module = MLP(
                in_features=get_space_shape(obs_space)[-1],
                out_features=hidden_sizes[-1],
                num_cells=hidden_sizes[:-1],
                activation_class=getattr(nn, activation_fn),
                device=device
            )
            out_features = hidden_sizes[-1]
        else:
            shared_module = nn.Identity()
            out_features = get_space_shape(obs_space)[-1]
        
        super().__init__(
            module=shared_module,
            in_keys=["observation"],
            out_keys=["shared"],
        )
        self.out_features = out_features

class Actor(TensorDictModule):
    def __init__(self, shared_module, act_space, hidden_sizes, activation_fn, device):
        act_space_shape = get_space_shape(act_space)
        if is_discrete_space(act_space):
            out_features = act_space_shape[-1]
        else:
            out_features = act_space_shape[-1] * 2

        actor_module = nn.Sequential(
            MLP(
                in_features=shared_module.out_features,
                out_features=out_features,
                num_cells=hidden_sizes,
                activation_class=getattr(nn, activation_fn),
                device=device
            ),
            NormalParamExtractor() if not is_discrete_space(act_space) else nn.Identity(),
        ).to(device)
        super().__init__(
            module=actor_module,
            in_keys=["shared"],
            out_keys=["loc", "scale"] if not is_discrete_space(act_space) else ["action_logits"],
        )

class Critic(TensorDictModule):
    def __init__(self, shared_module, hidden_sizes, activation_fn, device):
        critic_module = MLP(
            in_features=shared_module.out_features,
            out_features=1,
            num_cells=hidden_sizes,
            activation_class=getattr(nn, activation_fn),
            device=device
        ).to(device)
        super().__init__(
            module=critic_module,
            in_keys=["shared"],
            out_keys=["state_value"],
        )
