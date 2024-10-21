import torch
from torch import nn
from typing import Dict, Any, Optional
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives.value import GAE
from fancy_rl.algos.on_policy import OnPolicy
from fancy_rl.policy import Actor, Critic
from fancy_rl.projections import get_projection, BaseProjection
from fancy_rl.objectives import TRPLLoss
from fancy_rl.utils import is_discrete_space
from copy import deepcopy
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

class ProjectedActor(TensorDictModule):
    def __init__(self, raw_actor, old_actor, projection):
        combined_module = self.CombinedModule(raw_actor, old_actor, projection)
        super().__init__(
            module=combined_module,
            in_keys=raw_actor.in_keys,
            out_keys=raw_actor.out_keys
        )
        self.raw_actor = raw_actor
        self.old_actor = old_actor
        self.projection = projection

    class CombinedModule(nn.Module):
        def __init__(self, raw_actor, old_actor, projection):
            super().__init__()
            self.raw_actor = raw_actor
            self.old_actor = old_actor
            self.projection = projection

        def forward(self, tensordict):
            raw_params = self.raw_actor(tensordict)
            old_params = self.old_actor(tensordict)
            combined_params = TensorDict({**raw_params, **{f"old_{key}": value for key, value in old_params.items()}}, batch_size=tensordict.batch_size)
            projected_params = self.projection(combined_params)
            return projected_params

class TRPL(OnPolicy):
    def __init__(
        self,
        env_spec,
        loggers=None,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        actor_activation_fn="Tanh",
        critic_activation_fn="Tanh",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        projection_class="identity_projection",
        trust_region_coef=10.0,
        trust_region_bound_mean=0.1,
        trust_region_bound_cov=0.001,
        total_timesteps=1e6,
        eval_interval=2048,
        eval_deterministic=True,
        entropy_coef=0.01,
        critic_coef=0.5,
        normalize_advantage=False,
        device=None,
        env_spec_eval=None,
        eval_episodes=10,
        full_covariance=False,
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize environment to get observation and action space sizes
        self.env_spec = env_spec
        env = self.make_env()
        obs_space = env.observation_space
        act_space = env.action_space

        assert not is_discrete_space(act_space), "TRPL does not support discrete action spaces"

        self.critic = Critic(obs_space, critic_hidden_sizes, critic_activation_fn, device)
        self.raw_actor = Actor(obs_space, act_space, actor_hidden_sizes, actor_activation_fn, device, full_covariance=full_covariance)
        self.old_actor = Actor(obs_space, act_space, actor_hidden_sizes, actor_activation_fn, device, full_covariance=full_covariance)

        # Handle projection_class
        if isinstance(projection_class, str):
            projection_class = get_projection(projection_class)
        elif not issubclass(projection_class, BaseProjection):
            raise ValueError("projection_class must be a string or a subclass of BaseProjection")

        self.projection = projection_class(
            in_keys=["loc", "scale_tril", "old_loc", "old_scale_tril"] if full_covariance else ["loc", "scale", "old_loc", "old_scale"],
            out_keys=["loc", "scale_tril"] if full_covariance else ["loc", "scale"],
            mean_bound=trust_region_bound_mean,
            cov_bound=trust_region_bound_cov
        )

        self.actor = ProjectedActor(self.raw_actor, self.old_actor, self.projection)

        if full_covariance:
            distribution_class = torch.distributions.MultivariateNormal
            distribution_kwargs = {"loc": "loc", "scale_tril": "scale_tril"}
        else:
            distribution_class = torch.distributions.Normal
            distribution_kwargs = {"loc": "loc", "scale": "scale"}

        self.prob_actor = ProbabilisticActor(
            module=self.actor,
            distribution_class=distribution_class,
            return_log_prob=True,
            in_keys=distribution_kwargs,
        )

        self.trust_region_coef = trust_region_coef
        self.loss_module = TRPLLoss(
            actor_network=self.actor,
            old_actor_network=self.old_actor,
            critic_network=self.critic,
            projection=self.projection,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            trust_region_coef=trust_region_coef,
            normalize_advantage=normalize_advantage,
        )

        optimizers = {
            "actor": torch.optim.Adam(self.raw_actor.parameters(), lr=learning_rate),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        }

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
        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=gae_lambda,
            value_network=self.critic,
            average_gae=False,
        )

    def update_old_policy(self):
        self.old_actor.load_state_dict(self.raw_actor.state_dict())

    def post_update(self):
        self.update_old_policy()