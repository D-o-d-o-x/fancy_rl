import torch
from .base_projection import BaseProjection
from typing import Dict, Tuple
from torchrl.modules import TensorDictModule
from torchrl.distributions import TanhNormal, Delta

class W2Projection(BaseProjection):
    def __init__(self,
                 in_keys: list[str],
                 out_keys: list[str],
                 scale_prec: bool = False):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.scale_prec = scale_prec

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        projected_params = {}
        for key in policy_params.keys():
            if key.endswith('.loc'):
                mean = policy_params[key]
                old_mean = old_policy_params[key]
                std_key = key.replace('.loc', '.scale')
                std = policy_params[std_key]
                old_std = old_policy_params[std_key]
                
                projected_mean, projected_std = self._trust_region_projection(
                    mean, std, old_mean, old_std
                )
                
                projected_params[key] = projected_mean
                projected_params[std_key] = projected_std
            elif not key.endswith('.scale'):
                projected_params[key] = policy_params[key]
        
        return projected_params

    def _trust_region_projection(self, mean: torch.Tensor, std: torch.Tensor, 
                                 old_mean: torch.Tensor, old_std: torch.Tensor,
                                 eps: float = 1e-3, eps_cov: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_part, cov_part = self._gaussian_wasserstein_commutative(mean, std, old_mean, old_std)
        
        # Project mean
        mean_mask = mean_part > eps
        proj_mean = torch.where(mean_mask, 
                                old_mean + (mean - old_mean) * torch.sqrt(eps / mean_part)[..., None],
                                mean)
        
        # Project covariance
        cov_mask = cov_part > eps_cov
        eta = torch.ones_like(cov_part)
        eta[cov_mask] = torch.sqrt(cov_part[cov_mask] / eps_cov) - 1.
        eta = torch.clamp(eta, -0.9, float('inf'))  # Avoid negative values that could lead to invalid standard deviations
        
        proj_std = (std + eta[..., None] * old_std) / (1. + eta[..., None] + 1e-8)
        
        return proj_mean, proj_std

    def _gaussian_wasserstein_commutative(self, mean: torch.Tensor, std: torch.Tensor, 
                                          old_mean: torch.Tensor, old_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scale_prec:
            # Mahalanobis distance for mean
            mean_part = ((mean - old_mean) ** 2 / (old_std ** 2 + 1e-8)).sum(-1)
        else:
            # Euclidean distance for mean
            mean_part = ((mean - old_mean) ** 2).sum(-1)
        
        # W2 objective for covariance
        cov_part = (std ** 2 + old_std ** 2 - 2 * std * old_std).sum(-1)
        
        return mean_part, cov_part

    @classmethod
    def make(cls, in_keys: list[str], out_keys: list[str], **kwargs) -> 'W2Projection':
        return cls(in_keys=in_keys, out_keys=out_keys, **kwargs)