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
from copy import deepcopy

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
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize environment to get observation and action space sizes
        self.env_spec = env_spec
        env = self.make_env()
        obs_space = env.observation_space
        act_space = env.action_space

        self.critic = Critic(obs_space, critic_hidden_sizes, critic_activation_fn, device)
        actor_net = Actor(obs_space, act_space, actor_hidden_sizes, actor_activation_fn, device)

        # Handle projection_class
        if isinstance(projection_class, str):
            projection_class = get_projection(projection_class)
        elif not issubclass(projection_class, BaseProjection):
            raise ValueError("projection_class must be a string or a subclass of BaseProjection")

        self.projection = projection_class(
            in_keys=["loc", "scale"],
            out_keys=["loc", "scale"],
            trust_region_bound_mean=trust_region_bound_mean,
            trust_region_bound_cov=trust_region_bound_cov
        )

        self.actor = ProbabilisticActor(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
            distribution_class=torch.distributions.Normal,
            return_log_prob=True
        )
        self.old_actor = deepcopy(self.actor)

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
            "actor": torch.optim.Adam(self.actor.parameters(), lr=learning_rate),
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
        self.old_actor.load_state_dict(self.actor.state_dict())

    def project_policy(self, obs):
        with torch.no_grad():
            old_dist = self.old_actor(obs)
        new_dist = self.actor(obs)
        projected_params = self.projection.project(new_dist, old_dist)
        return projected_params

    def pre_update(self, tensordict):
        obs = tensordict["observation"]
        projected_dist = self.project_policy(obs)
        
        # Update tensordict with projected distribution parameters
        tensordict["projected_loc"] = projected_dist[0]
        tensordict["projected_scale"] = projected_dist[1]
        return tensordict

    def post_update(self):
        self.update_old_policy()
