from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Dict, List

class BaseProjection(nn.Module, ABC):
    def __init__(self, in_keys: List[str], out_keys: List[str], trust_region_coeff: float = 1.0, mean_bound: float = 0.01, cov_bound: float = 0.01, contextual_std: bool = True):
        super().__init__()
        self._validate_in_keys(in_keys)
        self._validate_out_keys(out_keys)
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.trust_region_coeff = trust_region_coeff
        self.mean_bound = mean_bound
        self.cov_bound = cov_bound
        self.full_cov = "scale_tril" in in_keys
        self.contextual_std = contextual_std

    def _validate_in_keys(self, keys: List[str]):
        valid_keys = {"loc", "scale", "scale_tril", "old_loc", "old_scale", "old_scale_tril"}
        if not set(keys).issubset(valid_keys):
            raise ValueError(f"Invalid in_keys: {keys}. Must be a subset of {valid_keys}")
        if "loc" not in keys or "old_loc" not in keys:
            raise ValueError("Both 'loc' and 'old_loc' must be included in in_keys")
        if ("scale" in keys) != ("old_scale" in keys) or ("scale_tril" in keys) != ("old_scale_tril" in keys):
            raise ValueError("in_keys must have matching 'scale'/'old_scale' or 'scale_tril'/'old_scale_tril'")

    def _validate_out_keys(self, keys: List[str]):
        valid_keys = {"loc", "scale", "scale_tril"}
        if not set(keys).issubset(valid_keys):
            raise ValueError(f"Invalid out_keys: {keys}. Must be a subset of {valid_keys}")
        if "loc" not in keys:
            raise ValueError("'loc' must be included in out_keys")
        if "scale" not in keys and "scale_tril" not in keys:
            raise ValueError("Either 'scale' or 'scale_tril' must be included in out_keys")

    @abstractmethod
    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def get_trust_region_loss(self, policy_params: Dict[str, torch.Tensor], proj_policy_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def forward(self, tensordict):
        policy_params = {}
        old_policy_params = {}
        
        for key in self.in_keys:
            if key not in tensordict:
                raise KeyError(f"Key '{key}' not found in tensordict. Available keys: {tensordict.keys()}")
            
            if key.startswith("old_"):
                old_policy_params[key[4:]] = tensordict[key]
            else:
                policy_params[key] = tensordict[key]

        projected_params = self.project(policy_params, old_policy_params)
        return projected_params

    def _calc_covariance(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.full_cov:
            return torch.diag_embed(params["scale"].pow(2))
        else:
            return torch.matmul(params["scale_tril"], params["scale_tril"].transpose(-1, -2))

    def _calc_scale_or_scale_tril(self, cov: torch.Tensor) -> torch.Tensor:
        if not self.full_cov:
            return torch.sqrt(cov.diagonal(dim1=-2, dim2=-1))
        else:
            return torch.linalg.cholesky(cov)