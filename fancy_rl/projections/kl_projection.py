import torch
import cpp_projection
import numpy as np
from .base_projection import BaseProjection
from tensordict.nn import TensorDictModule
from typing import Dict, Tuple, Any

MAX_EVAL = 1000

def get_numpy(tensor):
    return tensor.detach().cpu().numpy()

class KLProjection(BaseProjection):
    def __init__(self, in_keys: list[str], out_keys: list[str], trust_region_coeff: float = 1.0, mean_bound: float = 0.01, cov_bound: float = 0.01, contextual_std: bool = True):
        super().__init__(in_keys=in_keys, out_keys=out_keys, trust_region_coeff=trust_region_coeff, mean_bound=mean_bound, cov_bound=cov_bound, contextual_std=contextual_std)

    def project(self, policy_params: Dict[str, torch.Tensor], old_policy_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mean, scale_or_tril = policy_params["loc"], policy_params[self.in_keys[1]]
        old_mean, old_scale_or_tril = old_policy_params["loc"], old_policy_params[self.in_keys[1]]

        mean_part, cov_part = self._gaussian_kl((mean, scale_or_tril), (old_mean, old_scale_or_tril))

        if not self.contextual_std:
            scale_or_tril = scale_or_tril[:1]
            old_scale_or_tril = old_scale_or_tril[:1]
            cov_part = cov_part[:1]

        proj_mean = self._mean_projection(mean, old_mean, mean_part)
        proj_scale_or_tril = self._cov_projection(scale_or_tril, old_scale_or_tril, cov_part)

        if not self.contextual_std:
            proj_scale_or_tril = proj_scale_or_tril.expand(mean.shape[0], *proj_scale_or_tril.shape[1:])

        return {"loc": proj_mean, self.out_keys[1]: proj_scale_or_tril}

    def get_trust_region_loss(self, policy_params: Dict[str, torch.Tensor], proj_policy_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        mean, scale_or_tril = policy_params["loc"], policy_params[self.in_keys[1]]
        proj_mean, proj_scale_or_tril = proj_policy_params["loc"], proj_policy_params[self.out_keys[1]]
        kl = sum(self._gaussian_kl((mean, scale_or_tril), (proj_mean, proj_scale_or_tril)))
        return kl.mean() * self.trust_region_coeff

    def _gaussian_kl(self, p: Tuple[torch.Tensor, torch.Tensor], q: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, scale_or_tril = p
        mean_other, scale_or_tril_other = q
        k = mean.shape[-1]

        maha_part = 0.5 * self._maha(mean, mean_other, scale_or_tril_other)

        det_term = self._log_determinant(scale_or_tril)
        det_term_other = self._log_determinant(scale_or_tril_other)

        if self.full_cov:
            trace_part = self._torch_batched_trace_square(torch.linalg.solve_triangular(scale_or_tril_other, scale_or_tril, upper=False))
        else:
            trace_part = torch.sum((scale_or_tril / scale_or_tril_other) ** 2, dim=-1)

        cov_part = 0.5 * (trace_part - k + det_term_other - det_term)

        return maha_part, cov_part

    def _maha(self, x: torch.Tensor, y: torch.Tensor, scale_or_tril: torch.Tensor) -> torch.Tensor:
        diff = x - y
        if self.full_cov:
            return torch.sum(torch.square(torch.triangular_solve(diff.unsqueeze(-1), scale_or_tril, upper=False)[0].squeeze(-1)), dim=-1)
        else:
            return torch.sum(torch.square(diff / scale_or_tril), dim=-1)

    def _log_determinant(self, scale_or_tril: torch.Tensor) -> torch.Tensor:
        if self.full_cov:
            return 2 * torch.log(scale_or_tril.diagonal(dim1=-2, dim2=-1)).sum(-1)
        else:
            return 2 * torch.log(scale_or_tril).sum(-1)

    def _torch_batched_trace_square(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.pow(2), dim=(-2, -1))

    def _mean_projection(self, mean: torch.Tensor, old_mean: torch.Tensor, mean_part: torch.Tensor) -> torch.Tensor:
        return old_mean + (mean - old_mean) * torch.sqrt(self.mean_bound / (mean_part + 1e-8)).unsqueeze(-1)

    def _cov_projection(self, scale_or_tril: torch.Tensor, old_scale_or_tril: torch.Tensor, cov_part: torch.Tensor) -> torch.Tensor:
        if self.full_cov:
            cov = torch.matmul(scale_or_tril, scale_or_tril.transpose(-1, -2))
            old_cov = torch.matmul(old_scale_or_tril, old_scale_or_tril.transpose(-1, -2))
        else:
            cov = scale_or_tril.pow(2)
            old_cov = old_scale_or_tril.pow(2)

        mask = cov_part > self.cov_bound
        proj_scale_or_tril = torch.zeros_like(scale_or_tril)
        proj_scale_or_tril[~mask] = scale_or_tril[~mask]

        try:
            if mask.any():
                if self.full_cov:
                    proj_cov = KLProjectionGradFunctionCovOnly.apply(cov, scale_or_tril.detach(), old_scale_or_tril, self.cov_bound)
                    is_invalid = proj_cov.mean([-2, -1]).isnan() & mask
                    if is_invalid.any():
                        proj_scale_or_tril[is_invalid] = old_scale_or_tril[is_invalid]
                        mask &= ~is_invalid
                    proj_scale_or_tril[mask], failed_mask = torch.linalg.cholesky_ex(proj_cov[mask])
                    failed_mask = failed_mask.bool()
                    if failed_mask.any():
                        proj_scale_or_tril[failed_mask] = old_scale_or_tril[failed_mask]
                else:
                    proj_cov = KLProjectionGradFunctionDiagCovOnly.apply(cov, old_cov, self.cov_bound)
                    is_invalid = (proj_cov.mean(dim=-1).isnan() | proj_cov.mean(dim=-1).isinf() | (proj_cov.min(dim=-1).values < 0)) & mask
                    if is_invalid.any():
                        proj_scale_or_tril[is_invalid] = old_scale_or_tril[is_invalid]
                        mask &= ~is_invalid
                    proj_scale_or_tril[mask] = proj_cov[mask].sqrt()
        except Exception as e:
            import logging
            logging.error('Projection failed, taking old scale_or_tril for projection.')
            print("Projection failed, taking old scale_or_tril for projection.")
            proj_scale_or_tril = old_scale_or_tril
            raise e

        return proj_scale_or_tril


class KLProjectionGradFunctionCovOnly(torch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=MAX_EVAL):
        if not KLProjectionGradFunctionCovOnly.projection_op:
            KLProjectionGradFunctionCovOnly.projection_op = \
                cpp_projection.BatchedCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        cov, chol, old_chol, eps_cov = args

        batch_shape = cov.shape[0]
        dim = cov.shape[-1]

        cov_np = get_numpy(cov)
        chol_np = get_numpy(chol)
        old_chol_np = get_numpy(old_chol)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        p_op = KLProjectionGradFunctionCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_chol_np, chol_np, cov_np)

        return cov.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_cov, = grad_outputs

        d_cov_np = get_numpy(d_cov)
        d_cov_np = np.atleast_2d(d_cov_np)

        df_stds = projection_op.backward(d_cov_np)
        df_stds = np.atleast_2d(df_stds)

        df_stds = d_cov.new(df_stds)

        return df_stds, None, None, None


class KLProjectionGradFunctionDiagCovOnly(torch.autograd.Function):
    projection_op = None

    @staticmethod
    def get_projection_op(batch_shape, dim, max_eval=MAX_EVAL):
        if not KLProjectionGradFunctionDiagCovOnly.projection_op:
            KLProjectionGradFunctionDiagCovOnly.projection_op = \
                cpp_projection.BatchedDiagCovOnlyProjection(batch_shape, dim, max_eval=max_eval)
        return KLProjectionGradFunctionDiagCovOnly.projection_op

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        cov, old_cov, eps_cov = args

        batch_shape = cov.shape[0]
        dim = cov.shape[-1]

        cov_np = get_numpy(cov)
        old_cov_np = get_numpy(old_cov)
        eps = get_numpy(eps_cov) * np.ones(batch_shape)

        p_op = KLProjectionGradFunctionDiagCovOnly.get_projection_op(batch_shape, dim)
        ctx.proj = p_op

        proj_std = p_op.forward(eps, old_cov_np, cov_np)

        return cov.new(proj_std)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_std, = grad_outputs

        d_cov_np = get_numpy(d_std)
        d_cov_np = np.atleast_2d(d_cov_np)
        df_stds = projection_op.backward(d_cov_np)
        df_stds = np.atleast_2d(df_stds)

        return d_std.new(df_stds), None, None