import torch
from .base_projection import BaseProjection
from typing import Dict

class IdentityProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str]):
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The identity projection simply returns the new policy parameters without any modification
        return policy_params