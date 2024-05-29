# Fancy RL

Fancy RL is a minimalistic and efficient implementation of Proximal Policy Optimization (PPO) and Trust Region Policy Layers (TRPL) using primitives from [torchrl](https://pypi.org/project/torchrl/). Future plans include implementing Soft Actor-Critic (SAC). This library focuses on providing clean and understandable code while leveraging the powerful functionalities of torchrl.
We provide optional integration with wandb.

## Installation

Fancy RL requires Python 3.7-3.11. (TorchRL currently does not support Python 3.12)

```bash
pip install -e .
```

## Usage

Here's a basic example of how to train a PPO agent with Fancy RL:

```python
from fancy_rl.ppo import PPO
from fancy_rl.policy import Policy
import gymnasium as gym

def env_fn():
    return gym.make("CartPole-v1")

# Create policy
env = env_fn()
policy = Policy(env.observation_space, env.action_space)

# Create PPO instance with default config
ppo = PPO(policy=policy, env_fn=env_fn)

# Train the agent
ppo.train()
```

For a more complete function description and advanced usage, refer to `example/example.py`.

### Testing

To run the test suite:

```bash
pytest test/test_ppo.py
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the library.

## License

This project is licensed under the MIT License.
