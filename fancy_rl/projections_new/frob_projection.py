import torch
from .base_projection import BaseProjection
from typing import Dict

class FrobProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str], epsilon: float = 1e-3):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.epsilon = epsilon

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected_params = {}
        for key in policy_params.keys():
            old_param = old_policy_params[key]
            new_param = policy_params[key]
            diff = new_param - old_param
            norm = torch.norm(diff)
            if norm > self.epsilon:
                projected_param = old_param + (self.epsilon / norm) * diff
            else:
                projected_param = new_param
            projected_params[key] = projected_param
        return projected_params