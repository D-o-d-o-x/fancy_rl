try:
    import gym
    from gym.spaces import Discrete as GymDiscrete, MultiDiscrete as GymMultiDiscrete, MultiBinary as GymMultiBinary, Box as GymBox
except ImportError:
    gym = None

import gymnasium
from gymnasium.spaces import Discrete as GymnasiumDiscrete, MultiDiscrete as GymnasiumMultiDiscrete, MultiBinary as GymnasiumMultiBinary, Box as GymnasiumBox
from torchrl.data.tensor_specs import (
    DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec,
    BinaryDiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
)

def is_discrete_space(action_space):
    discrete_types = (
        GymDiscrete, GymMultiDiscrete, GymMultiBinary,
        GymnasiumDiscrete, GymnasiumMultiDiscrete, GymnasiumMultiBinary,
        DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec, BinaryDiscreteTensorSpec
    )
    continuous_types = (
        GymBox, GymnasiumBox, BoundedTensorSpec, UnboundedContinuousTensorSpec
    )
    
    if isinstance(action_space, discrete_types):
        return True
    elif isinstance(action_space, continuous_types):
        return False
    else:
        raise ValueError(f"Unsupported action space type: {type(action_space)}")

def get_space_shape(action_space):
    if gym is not None:
        discrete_types = (GymDiscrete, GymMultiDiscrete, GymMultiBinary)
        continuous_types = (GymBox,)
    else:
        discrete_types = ()
        continuous_types = ()

    discrete_types += (GymnasiumDiscrete, GymnasiumMultiDiscrete, GymnasiumMultiBinary,
                       DiscreteTensorSpec, OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec, BinaryDiscreteTensorSpec)
    continuous_types += (GymnasiumBox, BoundedTensorSpec, UnboundedContinuousTensorSpec)

    if isinstance(action_space, discrete_types):
        if isinstance(action_space, (GymDiscrete, GymnasiumDiscrete, DiscreteTensorSpec, OneHotDiscreteTensorSpec)):
            return (action_space.n,)
        elif isinstance(action_space, (GymMultiDiscrete, GymnasiumMultiDiscrete, MultiDiscreteTensorSpec)):
            return (sum(action_space.nvec),)
        elif isinstance(action_space, (GymMultiBinary, GymnasiumMultiBinary, BinaryDiscreteTensorSpec)):
            return (action_space.n,)
    elif isinstance(action_space, continuous_types):
        return action_space.shape
    
    raise ValueError(f"Unsupported action space type: {type(action_space)}")
