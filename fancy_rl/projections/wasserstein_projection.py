import torch
from .base_projection import BaseProjection
from typing import Dict, Tuple

def gaussian_wasserstein_commutative(policy, p: Tuple[torch.Tensor, torch.Tensor],
                                     q: Tuple[torch.Tensor, torch.Tensor], scale_prec=False) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, sqrt = p
    mean_other, sqrt_other = q

    mean_part = torch.sum(torch.square(mean - mean_other), dim=-1)

    cov = torch.matmul(sqrt, sqrt.transpose(-1, -2))
    cov_other = torch.matmul(sqrt_other, sqrt_other.transpose(-1, -2))

    if scale_prec:
        identity = torch.eye(mean.shape[-1], dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = torch.linalg.solve(sqrt_other, identity)
        c = sqrt_inv_other @ cov @ sqrt_inv_other
        cov_part = torch.trace(identity + c - 2 * sqrt_inv_other @ sqrt)
    else:
        cov_part = torch.trace(cov_other + cov - 2 * sqrt_other @ sqrt)

    return mean_part, cov_part

class WassersteinProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str], trust_region_coeff: float = 1.0, mean_bound: float = 0.01, cov_bound: float = 0.01, scale_prec: bool = False):
        super().__init__(in_keys=in_keys, out_keys=out_keys, trust_region_coeff=trust_region_coeff, mean_bound=mean_bound, cov_bound=cov_bound)
        self.scale_prec = scale_prec

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mean, sqrt = policy_params["loc"], policy_params["scale_tril"]
        old_mean, old_sqrt = old_policy_params["loc"], old_policy_params["scale_tril"]

        mean_part, cov_part = gaussian_wasserstein_commutative(None, (mean, sqrt), (old_mean, old_sqrt), self.scale_prec)

        proj_mean = self._mean_projection(mean, old_mean, mean_part)
        proj_sqrt = self._cov_projection(sqrt, old_sqrt, cov_part)

        return {"loc": proj_mean, "scale_tril": proj_sqrt}

    def get_trust_region_loss(self, policy_params: Dict[str, torch.Tensor], proj_policy_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean, sqrt = policy_params["loc"], policy_params["scale_tril"]
        proj_mean, proj_sqrt = proj_policy_params["loc"], proj_policy_params["scale_tril"]
        mean_part, cov_part = gaussian_wasserstein_commutative(None, (mean, sqrt), (proj_mean, proj_sqrt), self.scale_prec)
        w2 = mean_part + cov_part
        return w2.mean() * self.trust_region_coeff

    def _mean_projection(self, mean: torch.Tensor, old_mean: torch.Tensor, mean_part: torch.Tensor) -> torch.Tensor:
        diff = mean - old_mean
        norm = torch.norm(diff, dim=-1, keepdim=True)
        return torch.where(norm > self.mean_bound, old_mean + diff * self.mean_bound / norm, mean)

    def _cov_projection(self, sqrt: torch.Tensor, old_sqrt: torch.Tensor, cov_part: torch.Tensor) -> torch.Tensor:
        diff = sqrt - old_sqrt
        norm = torch.norm(diff, dim=(-2, -1), keepdim=True)
        return torch.where(norm > self.cov_bound, old_sqrt + diff * self.cov_bound / norm, sqrt)