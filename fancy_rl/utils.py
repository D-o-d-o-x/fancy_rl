import gymnasium
from gymnasium.spaces import Discrete as GymnasiumDiscrete, MultiDiscrete as GymnasiumMultiDiscrete, MultiBinary as GymnasiumMultiBinary, Box as GymnasiumBox
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec,
    BinaryDiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
)

try:
    import gym
    from gym.spaces import Discrete as GymDiscrete, MultiDiscrete as GymMultiDiscrete, MultiBinary as GymMultiBinary, Box as GymBox
    gym_available = True
except ImportError:
    gym_available = False

def is_discrete_space(action_space):
    discrete_types = (
        GymnasiumDiscrete, GymnasiumMultiDiscrete, GymnasiumMultiBinary,
        DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec, BinaryDiscreteTensorSpec
    )
    continuous_types = (
        GymnasiumBox, BoundedTensorSpec, UnboundedContinuousTensorSpec
    )
    
    if gym_available:
        discrete_types += (GymDiscrete, GymMultiDiscrete, GymMultiBinary)
        continuous_types += (GymBox,)
    
    if isinstance(action_space, discrete_types):
        return True
    elif isinstance(action_space, continuous_types):
        return False
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def get_space_shape(action_space):
    discrete_types = (GymnasiumDiscrete, GymnasiumMultiDiscrete, GymnasiumMultiBinary,
                      DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec, BinaryDiscreteTensorSpec)
    continuous_types = (GymnasiumBox, BoundedTensorSpec, UnboundedContinuousTensorSpec)

    if gym_available:
        discrete_types += (GymDiscrete, GymMultiDiscrete, GymMultiBinary)
        continuous_types += (GymBox,)

    if isinstance(action_space, discrete_types):
        if isinstance(action_space, (GymnasiumDiscrete, DiscreteTensorSpec, OneHotDiscreteTensorSpec)):
            return (action_space.n,)
        elif isinstance(action_space, (GymnasiumMultiDiscrete, MultiDiscreteTensorSpec)):
            return (sum(action_space.nvec),)
        elif isinstance(action_space, (GymnasiumMultiBinary, BinaryDiscreteTensorSpec)):
            return (action_space.n,)
        elif gym_available:
            if isinstance(action_space, GymDiscrete):
                return (action_space.n,)
            elif isinstance(action_space, GymMultiDiscrete):
                return (sum(action_space.nvec),)
            elif isinstance(action_space, GymMultiBinary):
                return (action_space.n,)
    elif isinstance(action_space, continuous_types):
        return action_space.shape
    
    raise ValueError(f"Unsupported action space type: {type(action_space)}")
