from abc import ABC, abstractmethod
import torch
from typing import Dict

class BaseProjection(ABC, torch.nn.Module):
    def __init__(self, in_keys: list[str], out_keys: list[str]):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    @abstractmethod
    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    def forward(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.project(policy_params, old_policy_params)