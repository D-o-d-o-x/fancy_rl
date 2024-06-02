<h1 align="center">
  <br>
  <img src='./fancy_rl.svg' width="250px">
  <br><br>
  <b>Fancy RL</b>
  <br><br>
</h1>

Fancy RL provides a minimalistic and efficient implementation of Proximal Policy Optimization (PPO) and Trust Region Policy Layers (TRPL) using primitives from [torchrl](https://pypi.org/project/torchrl/). This library focuses on providing clean, understandable code and reusable modules while leveraging the powerful functionalities of torchrl.

| :exclamation: This project is still WIP and not ready to be used. |
| ----------------------------------------------------------------- |

## Installation

Fancy RL requires Python 3.7-3.11. (TorchRL currently does not support Python 3.12)

```bash
pip install -e .
```

## Usage

Fancy RL provides two main components:

1. **Ready-to-use Classes for PPO / TRPL**: These classes allow you to quickly get started with reinforcement learning algorithms, enjoying the performance and hackability that comes with using TorchRL.

   ```python
   from fancy_rl import PPO, TRPL

   model = TRPL("CartPole-v1")

   model.train()
   ```

   For environments, you can pass any [gymnasium](https://gymnasium.farama.org/) or [Fancy Gym](https://alrhub.github.io/fancy_gym/) environment ID as a string, a function returning a gymnasium or torchrl environment, an already instantiated gymnasium or torchrl environment, or a dict that will be passed to gymnasium.make. Check 'example/example.py' for a more complete usage example.

2. **Additional Modules for TRPL**: Designed to integrate with torchrl's primitives-first approach, these modules are ideal for building custom algorithms with precise trust region projections.
   Oh, you want documentation for these? To bad... (TODO)

## Background on Trust Region Policy Layers (TRPL)

Trust region methods are essential in reinforcement learning for ensuring robust policy updates. Traditional methods like TRPO and PPO use approximations, which can sometimes violate constraints or fail to find optimal solutions. To address these issues, TRPL provides differentiable neural network layers that enforce trust regions through closed-form projections for deep Gaussian policies. These layers formalize trust regions individually for each state and complement existing reinforcement learning algorithms.

The TRPL implementation in Fancy RL includes projections based on the Kullback-Leibler divergence, the Wasserstein L2 distance, and the Frobenius norm for Gaussian distributions. This approach achieves similar or better results than existing methods while being less sensitive to specific implementation choices.

## Testing

To run the test suite:

```bash
pytest test/test_ppo.py
```

## TODO

- [ ] Better / more logging
- [ ] Test / Benchmark PPO
- [ ] Refactor Modules for TRPL
- [ ] Get TRPL working
- [ ] Test / Benchmark TRPL
- [ ] Make contextual covariance optional
- [ ] Allow full-cov via chol
- [ ] Write docs / extend README
- [ ] (Implement SAC?)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the library.

## License

This project is licensed under the MIT License.
