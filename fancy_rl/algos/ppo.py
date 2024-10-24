import torch
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from fancy_rl.algos.on_policy import OnPolicy
from fancy_rl.policy import Actor, Critic
from fancy_rl.utils import is_discrete_space

class PPO(OnPolicy):
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
        total_timesteps=1e6,
        eval_interval=2048,
        eval_deterministic=True,
        entropy_coef=0.01,
        critic_coef=0.5,
        normalize_advantage=True,
        clip_range=0.2,
        device=None,
        env_spec_eval=None,
        eval_episodes=10,
        full_covariance=False,
    ):
        self.clip_range = clip_range

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Initialize environment to get observation and action space sizes
        self.env_spec = env_spec
        env = self.make_env()
        obs_space = env.observation_space
        act_space = env.action_space

        self.discrete = is_discrete_space(act_space)

        self.critic = Critic(obs_space, critic_hidden_sizes, critic_activation_fn, device)
        self.actor = Actor(obs_space, act_space, actor_hidden_sizes, actor_activation_fn, device, full_covariance=full_covariance)

        if self.discrete:
            distribution_class = torch.distributions.Categorical
            distribution_kwargs = {"logits": "action_logits"}
        else:
            if full_covariance:
                distribution_class = torch.distributions.MultivariateNormal
                in_keys = ["loc", "scale_tril"]
            else:
                distribution_class = torch.distributions.Normal
                in_keys = ["loc", "scale"]

            self.prob_actor = ProbabilisticActor(
                module=self.actor,
                distribution_class=distribution_class,
                return_log_prob=True,
                in_keys=in_keys,
                out_keys=["action"]
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

        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_range,
            loss_critic_type='l2',
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            normalize_advantage=self.normalize_advantage,
        )