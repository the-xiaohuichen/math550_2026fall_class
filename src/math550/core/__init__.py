"""Small cross-topic numerical and policy-optimization primitives."""

from .alignment import (
    GroupAdvantageResult,
    PPOTokenSurrogate,
    PolicyGradientResult,
    aggregate_token_losses,
    compute_group_advantages,
    masked_mean,
    policy_gradient_loss,
    ppo_token_surrogate,
    reference_kl_estimate,
)
from .numerics import logit, sigmoid, soft_threshold

__all__ = [
    "GroupAdvantageResult",
    "PPOTokenSurrogate",
    "PolicyGradientResult",
    "aggregate_token_losses",
    "compute_group_advantages",
    "logit",
    "masked_mean",
    "policy_gradient_loss",
    "ppo_token_surrogate",
    "reference_kl_estimate",
    "sigmoid",
    "soft_threshold",
]
