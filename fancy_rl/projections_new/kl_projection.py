import torch
from typing import Dict, List
from .base_projection import BaseProjection

class KLProjection(BaseProjection):
    def __init__(
        self,
        in_keys: List[str] = ["mean", "std"],
        out_keys: List[str] = ["projected_mean", "projected_std"],
        epsilon: float = 0.1
    ):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.epsilon = epsilon

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_mean, new_std = policy_params["mean"], policy_params["std"]
        old_mean, old_std = old_policy_params["mean"], old_policy_params["std"]

        diff = new_mean - old_mean
        std_diff = new_std - old_std

        kl = 0.5 * (torch.sum(torch.square(diff / old_std), dim=-1) +
                    torch.sum(torch.square(std_diff / old_std), dim=-1) -
                    new_mean.shape[-1] +
                    torch.sum(torch.log(new_std / old_std), dim=-1))

        factor = torch.sqrt(self.epsilon / (kl + 1e-8))
        factor = torch.clamp(factor, max=1.0)

        projected_mean = old_mean + factor.unsqueeze(-1) * diff
        projected_std = old_std + factor.unsqueeze(-1) * std_diff

        return {"mean": projected_mean, "std": projected_std}