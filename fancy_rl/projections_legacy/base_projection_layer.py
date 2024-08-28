from typing import Any, Dict, Optional, Type, Union, Tuple, final

import torch as th

from fancy_rl.norm import *

class BaseProjectionLayer(object):
    def __init__(self,
                 mean_bound: float = 0.03,
                 cov_bound: float = 1e-3,
                 trust_region_coeff: float = 1.0,
                 scale_prec: bool = False,
                 ):
        self.mean_bound = mean_bound
        self.cov_bound = cov_bound
        self.trust_region_coeff = trust_region_coeff
        self.scale_prec = scale_prec
        self.mean_eq = False

    def __call__(self, p, q, **kwargs):
        return self._projection(p, q, eps=self.mean_bound, eps_cov=self.cov_bound, beta=None, **kwargs)

    @final
    def _projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, beta: th.Tensor, **kwargs):
        return self._trust_region_projection(
            p, q, eps, eps_cov, **kwargs)

    def _trust_region_projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, **kwargs):
        """
        Hook for implementing the specific trust region projection
        Args:
            p: current distribution
            q: old distribution
            eps: mean trust region bound
            eps_cov: covariance trust region bound
            **kwargs:

        Returns:
            projected
        """
        return p

    def get_trust_region_loss(self, p, proj_p):
        # p:
        #   predicted distribution from network output
        # proj_p:
        #   projected distribution

        proj_mean, proj_chol = get_mean_and_chol(proj_p)
        p_target = new_dist_like(p, proj_mean, proj_chol)
        kl_diff = self.trust_region_value(p, p_target)

        kl_loss = kl_diff.mean()

        return kl_loss * self.trust_region_coeff

    def trust_region_value(self, p, q):
        """
        Computes the KL divergence between two Gaussian distributions p and q_values.
        Returns:
            full kl divergence
        """
        return kl_divergence(p, q)

    def new_dist_like(self, orig_p, mean, cov_cholesky):
        assert isinstance(orig_p, Distribution)
        p = orig_p.distribution
        if isinstance(p, th.distributions.Normal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Normal(mean, cov_cholesky)
        elif isinstance(p, th.distributions.Independent):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Independent(
                th.distributions.Normal(mean, cov_cholesky), 1)
        elif isinstance(p, th.distributions.MultivariateNormal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.MultivariateNormal(
                mean, scale_tril=cov_cholesky)
        else:
            raise Exception('Dist-Type not implemented (of sb3 dist)')
        return p_out

def entropy_inequality_projection(p: th.distributions.Normal,
                                  beta: Union[float, th.Tensor]):
    """
    Projects std to satisfy an entropy INEQUALITY constraint.
    Args:
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p.mean, p.stddev
    k = std.shape[-1]
    batch_shape = std.shape[:-2]

    ent = p.entropy()
    mask = ent < beta

    # if nothing has to be projected skip computation
    if (~mask).all():
        return p

    alpha = th.ones(batch_shape, dtype=std.dtype, device=std.device)
    alpha[mask] = th.exp((beta[mask] - ent[mask]) / k)

    proj_std = th.einsum('ijk,i->ijk', std, alpha)
    new_mean, new_std = mean, th.where(mask[..., None, None], proj_std, std)
    return th.distributions.Normal(new_mean, new_std)


def entropy_equality_projection(p: th.distributions.Normal,
                                beta: Union[float, th.Tensor]):
    """
    Projects std to satisfy an entropy EQUALITY constraint.
    Args:
        p: current distribution
        beta: target entropy for EACH std or general bound for all stds

    Returns:
        projected std that satisfies the entropy bound
    """
    mean, std = p.mean, p.stddev
    k = std.shape[-1]

    ent = p.entropy()
    alpha = th.exp((beta - ent) / k)
    proj_std = th.einsum('ijk,i->ijk', std, alpha)
    new_mean, new_std = mean, proj_std
    return th.distributions.Normal(new_mean, new_std)


def mean_projection(mean: th.Tensor, old_mean: th.Tensor, maha: th.Tensor, eps: th.Tensor):
    """
    Projects the mean based on the Mahalanobis objective and trust region.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound

    Returns:
        projected mean that satisfies the trust region
    """
    batch_shape = mean.shape[:-1]
    mask = maha > eps

    ################################################################################################################
    # mean projection maha

    # if nothing has to be projected skip computation
    if mask.any():
        omega = th.ones(batch_shape, dtype=mean.dtype, device=mean.device)
        omega[mask] = th.sqrt(maha[mask] / eps) - 1.
        omega = th.max(-omega, omega)[..., None]

        m = (mean + omega * old_mean) / (1 + omega + 1e-16)
        proj_mean = th.where(mask[..., None], m, mean)
    else:
        proj_mean = mean

    return proj_mean


def mean_equality_projection(mean: th.Tensor, old_mean: th.Tensor, maha: th.Tensor, eps: th.Tensor):
    """
    Projections the mean based on the Mahalanobis objective and trust region for an EQUALITY constraint.
    Args:
        mean: current mean vectors
        old_mean: old mean vectors
        maha: Mahalanobis distance between the two mean vectors
        eps: trust region bound
    Returns:
        projected mean that satisfies the trust region
    """

    maha[maha == 0] += 1e-16
    omega = th.sqrt(maha / eps) - 1.
    omega = omega[..., None]

    proj_mean = (mean + omega * old_mean) / (1 + omega + 1e-16)

    return proj_mean


class ITPALExceptionLayer(BaseProjectionLayer):
    def __init__(self,
                 *args, **kwargs
                 ):
        raise Exception('To be able to use KL projections, ITPAL must be installed: https://github.com/ALRhub/ITPAL.')