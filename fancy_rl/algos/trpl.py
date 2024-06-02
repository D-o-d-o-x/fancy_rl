import torch
from torchrl.modules import ActorValueOperator, ProbabilisticActor
from torchrl.objectives.value.advantages import GAE
from fancy_rl.algos.on_policy import OnPolicy
from fancy_rl.policy import Actor, Critic, SharedModule
from fancy_rl.objectives import TRPLLoss

class TRPL(OnPolicy):
    def __init__(
        self,
        env_spec,
        loggers=None,
        actor_hidden_sizes=[64, 64],
        critic_hidden_sizes=[64, 64],
        actor_activation_fn="Tanh",
        critic_activation_fn="Tanh",
        shared_stem_sizes=[64],
        proj_layer_type=None,
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
        trust_region_coef=10.0,
        normalize_advantage=True,
        device=None,
        env_spec_eval=None,
        eval_episodes=10,
    ):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trust_region_layer = None # from proj_layer_type

        # Initialize environment to get observation and action space sizes
        self.env_spec = env_spec
        env = self.make_env()
        obs_space = env.observation_space
        act_space = env.action_space

        # Define the shared, actor, and critic modules
        self.shared_module = SharedModule(obs_space, shared_stem_sizes, actor_activation_fn, device)
        self.raw_actor = Actor(self.shared_module, act_space, actor_hidden_sizes, actor_activation_fn, device)
        self.critic = Critic(self.shared_module, critic_hidden_sizes, critic_activation_fn, device)

        # Perfrom projection
        self.actor = self.raw_actor # TODO: Project

        # Combine into an ActorValueOperator
        self.ac_module = ActorValueOperator(
            self.shared_module,
            self.actor,
            self.critic
        )

        # Define the policy as a ProbabilisticActor
        policy = ProbabilisticActor(
            module=self.ac_module.get_policy_operator(),
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=torch.distributions.Normal,
            return_log_prob=True
        )

        optimizers = {
            "actor": torch.optim.Adam(self.actor.parameters(), lr=learning_rate),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        }

        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=False,
        )

        self.loss_module = TRPLLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            trust_region_layer=self.trust_region_layer,
            loss_critic_type='MSELoss',
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            trust_region_coef=self.trust_region_coef,
            normalize_advantage=self.normalize_advantage,
        )

        super().__init__(
            policy=policy,
            env_spec=env_spec,
            loggers=loggers,
            optimizers=optimizers,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            total_timesteps=total_timesteps,
            eval_interval=eval_interval,
            eval_deterministic=eval_deterministic,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=normalize_advantage,
            clip_range=clip_range,
            device=device,
            env_spec_eval=env_spec_eval,
            eval_episodes=eval_episodes,
        )