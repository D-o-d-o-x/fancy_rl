import torch
from .base_projection import BaseProjection
from tensordict.nn import TensorDictModule
from typing import Dict

class FrobeniusProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str], trust_region_coeff: float = 1.0, mean_bound: float = 0.01, cov_bound: float = 0.01, scale_prec: bool = False, contextual_std: bool = True):
        super().__init__(in_keys=in_keys, out_keys=out_keys, trust_region_coeff=trust_region_coeff, mean_bound=mean_bound, cov_bound=cov_bound, contextual_std=contextual_std)
        self.scale_prec = scale_prec

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mean = policy_params["loc"]
        old_mean = old_policy_params["loc"]

        cov = self._calc_covariance(policy_params)
        old_cov = self._calc_covariance(old_policy_params)

        mean_part, cov_part = self._gaussian_frobenius((mean, cov), (old_mean, old_cov))

        proj_mean = self._mean_projection(mean, old_mean, mean_part)
        proj_cov = self._cov_projection(cov, old_cov, cov_part)

        scale_or_scale_tril = self._calc_scale_or_scale_tril(proj_cov)
        return {"loc": proj_mean, self.out_keys[1]: scale_or_scale_tril}

    def get_trust_region_loss(self, policy_params: Dict[str, torch.Tensor], proj_policy_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean = policy_params["loc"]
        proj_mean = proj_policy_params["loc"]

        cov = self._calc_covariance(policy_params)
        proj_cov = self._calc_covariance(proj_policy_params)

        mean_diff = torch.sum(torch.square(mean - proj_mean), dim=-1)
        cov_diff = torch.sum(torch.square(cov - proj_cov), dim=(-2, -1))

        return (mean_diff + cov_diff).mean() * self.trust_region_coeff

    def _gaussian_frobenius(self, p, q):
        mean, cov = p
        old_mean, old_cov = q

        if self.scale_prec:
            prec_old = torch.inverse(old_cov)
            mean_part = torch.sum(torch.matmul(mean - old_mean, prec_old) * (mean - old_mean), dim=-1)
            cov_part = torch.sum(prec_old * cov, dim=(-2, -1)) - torch.logdet(torch.matmul(prec_old, cov)) - mean.shape[-1]
        else:
            mean_part = torch.sum(torch.square(mean - old_mean), dim=-1)
            cov_part = torch.sum(torch.square(cov - old_cov), dim=(-2, -1))

        return mean_part, cov_part

    def _mean_projection(self, mean: torch.Tensor, old_mean: torch.Tensor, mean_part: torch.Tensor) -> torch.Tensor:
        diff = mean - old_mean
        norm = torch.sqrt(mean_part)
        return torch.where(norm > self.mean_bound, old_mean + diff * self.mean_bound / norm.unsqueeze(-1), mean)

    def _cov_projection(self, cov: torch.Tensor, old_cov: torch.Tensor, cov_part: torch.Tensor) -> torch.Tensor:
        batch_shape = cov.shape[:-2]
        cov_mask = cov_part > self.cov_bound

        eta = torch.ones(batch_shape, dtype=cov.dtype, device=cov.device)
        eta[cov_mask] = torch.sqrt(cov_part[cov_mask] / self.cov_bound) - 1.
        eta = torch.max(-eta, eta)

        new_cov = (cov + torch.einsum('i,ijk->ijk', eta, old_cov)) / (1. + eta + 1e-16)[..., None, None]
        proj_cov = torch.where(cov_mask[..., None, None], new_cov, cov)

        return proj_cov