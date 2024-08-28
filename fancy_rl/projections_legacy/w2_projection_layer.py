import numpy as np
import torch as th
from typing import Tuple, Any

from ..misc.norm import mahalanobis

from .base_projection_layer import BaseProjectionLayer, mean_projection

from ..misc.norm import mahalanobis, _batch_trace
from ..misc.distTools import get_diag_cov_vec, get_mean_and_chol,  get_mean_and_sqrt, get_cov, has_diag_cov

from stable_baselines3.common.distributions import Distribution


class WassersteinProjectionLayer(BaseProjectionLayer):
    """
    Stolen from Fabian's Code (Public Version)
    """

    def _trust_region_projection(self, p, q, eps: th.Tensor, eps_cov: th.Tensor, **kwargs):
        """
        Runs commutative Wasserstein projection layer and constructs sqrt of covariance
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
            eps: (modified) kl bound/ kl bound for mean part
            eps_cov: (modified) kl bound for cov part
            **kwargs:

        Returns:
            mean, cov sqrt
        """

        mean, sqrt = get_mean_and_sqrt(p, expand=True)
        old_mean, old_sqrt = get_mean_and_sqrt(q, expand=True)
        batch_shape = mean.shape[:-1]

        ####################################################################################################################
        # precompute mean and cov part of W2, which are used for the projection.
        # Both parts differ based on precision scaling.
        # If activated, the mean part is the maha distance and the cov has a more complex term in the inner parenthesis.
        mean_part, cov_part = gaussian_wasserstein_commutative(
            p, q, self.scale_prec)

        ####################################################################################################################
        # project mean (w/ or w/o precision scaling)
        proj_mean = mean_projection(mean, old_mean, mean_part, eps)

        ####################################################################################################################
        # project covariance (w/ or w/o precision scaling)

        cov_mask = cov_part > eps_cov

        if cov_mask.any():
            # gradient issue with ch.where, it executes both paths and gives NaN gradient.
            eta = th.ones(batch_shape, dtype=sqrt.dtype, device=sqrt.device)
            eta[cov_mask] = th.sqrt(cov_part[cov_mask] / eps_cov) - 1.
            eta = th.max(-eta, eta)

            new_sqrt = (sqrt + th.einsum('i,ijk->ijk', eta, old_sqrt)
                        ) / (1. + eta + 1e-16)[..., None, None]
            proj_sqrt = th.where(cov_mask[..., None, None], new_sqrt, sqrt)
        else:
            proj_sqrt = sqrt

        if has_diag_cov(p):
            proj_sqrt = th.diagonal(proj_sqrt, dim1=-2, dim2=-1)

        proj_p = self.new_dist_like(p, proj_mean, proj_sqrt)
        return proj_p

    def trust_region_value(self, p, q):
        """
        Computes the Wasserstein distance between two Gaussian distributions p and q.
        Args:
            policy: policy instance
            p: current distribution
            q: old distribution
        Returns:
            mean and covariance part of Wasserstein distance
        """
        mean_part, cov_part = gaussian_wasserstein_commutative(
            p, q, scale_prec=self.scale_prec)
        return mean_part + cov_part

    def get_trust_region_loss(self, p, proj_p):
        # p:
        #   predicted distribution from network output
        # proj_p:
        #   projected distribution

        proj_mean, proj_sqrt = get_mean_and_sqrt(proj_p)
        p_target = self.new_dist_like(p, proj_mean, proj_sqrt)
        kl_diff = self.trust_region_value(p, p_target)

        kl_loss = kl_diff.mean()

        return kl_loss * self.trust_region_coeff

    def new_dist_like(self, orig_p, mean, cov_sqrt):
        assert isinstance(orig_p, Distribution)
        p = orig_p.distribution
        if isinstance(p, th.distributions.Normal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Normal(mean, cov_sqrt)
        elif isinstance(p, th.distributions.Independent):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Independent(
                th.distributions.Normal(mean, cov_sqrt), 1)
        elif isinstance(p, th.distributions.MultivariateNormal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.MultivariateNormal(
                mean, scale_tril=cov_sqrt, validate_args=False)
        else:
            raise Exception('Dist-Type not implemented (of sb3 dist)')
        p_out.cov_sqrt = cov_sqrt
        return p_out


def gaussian_wasserstein_commutative(p, q, scale_prec=False) -> Tuple[th.Tensor, th.Tensor]:
    """
    Compute mean part and cov part of W_2(p || q_values) with p,q_values ~ N(y, SS).
    This version DOES assume commutativity of both distributions, i.e. covariance matrices.
    This is less general and assumes both distributions are somewhat close together.
    When scale_prec is true scale both distributions with old precision matrix.
    Args:
        policy: current policy
        p: mean and sqrt of gaussian p
        q: mean and sqrt of gaussian q_values
        scale_prec: scale objective by old precision matrix.
                    This penalizes directions based on old uncertainty/covariance.
    Returns: mean part of W2, cov part of W2
    """
    mean, sqrt = get_mean_and_sqrt(p, expand=True)
    mean_other, sqrt_other = get_mean_and_sqrt(q, expand=True)

    if scale_prec:
        # maha objective for mean
        mean_part = mahalanobis(mean, mean_other, sqrt_other)
    else:
        # euclidean distance for mean
        # mean_part = ch.norm(mean_other - mean, ord=2, axis=1) ** 2
        mean_part = ((mean_other - mean) ** 2).sum(1)

    cov = get_cov(p)
    if scale_prec and False:
        # cov constraint scaled with precision of old dist
        batch_dim, dim = mean.shape

        identity = th.eye(dim, dtype=sqrt.dtype, device=sqrt.device)
        sqrt_inv_other = th.linalg.solve(sqrt_other, identity)
        c = sqrt_inv_other @ cov @ sqrt_inv_other

        cov_part = _batch_trace(
            identity + c - 2 * sqrt_inv_other @ sqrt)

    else:
        # W2 objective for cov assuming normal W2 objective for mean
        cov_other = get_cov(q)
        cov_part = _batch_trace(
            cov_other + cov - 2 * th.bmm(sqrt_other, sqrt))

    return mean_part, cov_part
