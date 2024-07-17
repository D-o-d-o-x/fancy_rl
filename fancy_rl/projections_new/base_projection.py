import torch
from torchrl.modules import TensorDictModule
from typing import List, Dict, Any

class BaseProjection(TensorDictModule):
    def __init__(
        self,
        in_keys: List[str],
        out_keys: List[str],
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def forward(self, tensordict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mean, std = self.in_keys
        projected_mean, projected_std = self.out_keys

        old_mean = tensordict[mean]
        old_std = tensordict[std]

        new_mean = tensordict.get(projected_mean, old_mean)
        new_std = tensordict.get(projected_std, old_std)

        projected_params = self.project(
            {"mean": new_mean, "std": new_std},
            {"mean": old_mean, "std": old_std}
        )

        tensordict[projected_mean] = projected_params["mean"]
        tensordict[projected_std] = projected_params["std"]

        return tensordict

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the project method")

    @classmethod
    def make(cls, projection_type: str, **kwargs: Any) -> 'BaseProjection':
        if projection_type == "kl":
            from .kl_projection import KLProjection
            return KLProjection(**kwargs)
        elif projection_type == "w2":
            from .w2_projection import W2Projection
            return W2Projection(**kwargs)
        elif projection_type == "frob":
            from .frob_projection import FrobProjection
            return FrobProjection(**kwargs)
        elif projection_type == "identity":
            from .identity_projection import IdentityProjection
            return IdentityProjection(**kwargs)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")