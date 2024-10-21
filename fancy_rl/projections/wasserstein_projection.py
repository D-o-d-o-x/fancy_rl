import torch
from .base_projection import BaseProjection
from tensordict.nn import TensorDictModule
from typing import Dict, Tuple

def scale_tril_to_sqrt(scale_tril: torch.Tensor) -> torch.Tensor:
    """
    'Converts' scale_tril to scale_sqrt.
    
    For Wasserstein distance, we need the matrix square root, not the Cholesky decomposition.
    But since both are lower triangular, we can treat the Cholesky decomposition as if it were the matrix square root.
    """
    return scale_tril

def gaussian_wasserstein_commutative(policy, p: Tuple[torch.Tensor, torch.Tensor],
                                     q: Tuple[torch.Tensor, torch.Tensor], scale_prec=False) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, scale_or_sqrt = p
    mean_other, scale_or_sqrt_other = q

    mean_part = torch.sum(torch.square(mean - mean_other), dim=-1)

    if scale_or_sqrt.dim() == mean.dim():  # Diagonal case
        cov = scale_or_sqrt.pow(2)
        cov_other = scale_or_sqrt_other.pow(2)
        if scale_prec:
            identity = torch.eye(mean.shape[-1], dtype=scale_or_sqrt.dtype, device=scale_or_sqrt.device)
            sqrt_inv_other = 1 / scale_or_sqrt_other
            c = sqrt_inv_other.pow(2) * cov
            cov_part = torch.sum(identity + c - 2 * sqrt_inv_other * scale_or_sqrt, dim=-1)
        else:
            cov_part = torch.sum(cov_other + cov - 2 * scale_or_sqrt_other * scale_or_sqrt, dim=-1)
    else:  # Full covariance case
        # Note: scale_or_sqrt is treated as the matrix square root, not Cholesky decomposition
        cov = torch.matmul(scale_or_sqrt, scale_or_sqrt.transpose(-1, -2))
        cov_other = torch.matmul(scale_or_sqrt_other, scale_or_sqrt_other.transpose(-1, -2))
        if scale_prec:
            identity = torch.eye(mean.shape[-1], dtype=scale_or_sqrt.dtype, device=scale_or_sqrt.device)
            sqrt_inv_other = torch.linalg.solve(scale_or_sqrt_other, identity)
            c = sqrt_inv_other @ cov @ sqrt_inv_other.transpose(-1, -2)
            cov_part = torch.trace(identity + c - 2 * sqrt_inv_other @ scale_or_sqrt)
        else:
            cov_part = torch.trace(cov_other + cov - 2 * scale_or_sqrt_other @ scale_or_sqrt)

    return mean_part, cov_part

class WassersteinProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str], trust_region_coeff: float = 1.0, mean_bound: float = 0.01, cov_bound: float = 0.01, scale_prec: bool = False, contextual_std: bool = True):
        super().__init__(in_keys=in_keys, out_keys=out_keys, trust_region_coeff=trust_region_coeff, mean_bound=mean_bound, cov_bound=cov_bound, contextual_std=contextual_std)
        self.scale_prec = scale_prec

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mean = policy_params["loc"]
        old_mean = old_policy_params["loc"]
        scale_or_sqrt = scale_tril_to_sqrt(policy_params[self.in_keys[1]])
        old_scale_or_sqrt = scale_tril_to_sqrt(old_policy_params[self.in_keys[1]])

        mean_part, cov_part = gaussian_wasserstein_commutative(None, (mean, scale_or_sqrt), (old_mean, old_scale_or_sqrt), self.scale_prec)

        proj_mean = self._mean_projection(mean, old_mean, mean_part)
        proj_scale_or_sqrt = self._cov_projection(scale_or_sqrt, old_scale_or_sqrt, cov_part)

        return {"loc": proj_mean, self.out_keys[1]: proj_scale_or_sqrt}

    def get_trust_region_loss(self, policy_params: Dict[str, torch.Tensor], proj_policy_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean = policy_params["loc"]
        proj_mean = proj_policy_params["loc"]
        scale_or_sqrt = scale_tril_to_sqrt(policy_params[self.in_keys[1]])
        proj_scale_or_sqrt = scale_tril_to_sqrt(proj_policy_params[self.out_keys[1]])
        mean_part, cov_part = gaussian_wasserstein_commutative(None, (mean, scale_or_sqrt), (proj_mean, proj_scale_or_sqrt), self.scale_prec)
        w2 = mean_part + cov_part
        return w2.mean() * self.trust_region_coeff

    def _mean_projection(self, mean: torch.Tensor, old_mean: torch.Tensor, mean_part: torch.Tensor) -> torch.Tensor:
        diff = mean - old_mean
        norm = torch.sqrt(mean_part)
        return torch.where(norm > self.mean_bound, old_mean + diff * self.mean_bound / norm.unsqueeze(-1), mean)

    def _cov_projection(self, scale_or_sqrt: torch.Tensor, old_scale_or_sqrt: torch.Tensor, cov_part: torch.Tensor) -> torch.Tensor:
        if scale_or_sqrt.dim() == old_scale_or_sqrt.dim() == 2:  # Diagonal case
            diff = scale_or_sqrt - old_scale_or_sqrt
            norm = torch.sqrt(cov_part)
            return torch.where(norm > self.cov_bound, old_scale_or_sqrt + diff * self.cov_bound / norm.unsqueeze(-1), scale_or_sqrt)
        else:  # Full covariance case
            diff = scale_or_sqrt - old_scale_or_sqrt
            norm = torch.norm(diff, dim=(-2, -1), keepdim=True)
            return torch.where(norm > self.cov_bound, old_scale_or_sqrt + diff * self.cov_bound / norm, scale_or_sqrt)