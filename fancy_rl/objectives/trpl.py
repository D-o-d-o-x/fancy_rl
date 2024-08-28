from __future__ import annotations

import contextlib

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl.objectives.common import LossModule

from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)

from torchrl.objectives.ppo import PPOLoss
from fancy_rl.projections import get_projection

class TRPLLoss(PPOLoss):
    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        old_actor_network: ProbabilisticTensorDictSequential,
        critic_network: TensorDictModule,
        projection: any,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        trust_region_coef: float = 10.0,
        normalize_advantage: bool = False,
        **kwargs,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            normalize_advantage=normalize_advantage,
            **kwargs,
        )
        self.old_actor_network = old_actor_network
        self.projection = projection
        self.trust_region_coef = trust_region_coef

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "tr_loss"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.critic_coef:
                keys.append("loss_critic")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def _trust_region_loss(self, tensordict):
        old_distribution = self.old_actor_network(tensordict)
        raw_distribution = self.actor_network(tensordict)
        return self.projection(self.actor_network, raw_distribution, old_distribution)

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        trust_region_loss_unscaled = self._trust_region_loss(tensordict)
        
        with torch.no_grad():
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        surrogate_gain = log_weight.exp() * advantage
        trust_region_loss = trust_region_loss_unscaled * self.trust_region_coef

        loss = -surrogate_gain + trust_region_loss
        td_out = TensorDict({"loss_objective": loss}, batch_size=[])
        td_out.set("tr_loss", trust_region_loss)

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out
