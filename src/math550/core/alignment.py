"""Shared policy-optimization primitives for control RL and language-model RL.

The functions in this file deliberately expose choices that are often hidden
inside an ``RLTrainer``: the baseline, advantage normalizer, importance ratio,
clipping rule, and loss denominator.  Keeping those choices orthogonal makes
GRPO, Dr. GRPO, rejection fine-tuning, MaxRL, and GSPO small configurations of
one score-function estimator rather than unrelated named algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


Baseline = Literal["mean", "none"]
AdvantageNormalizer = Literal["std", "mean", "none"]
ImportanceMethod = Literal["none", "noclip", "grpo", "gspo", "cispo"]
LossNormalization = Literal["sequence", "token", "constant"]


@dataclass(frozen=True)
class GroupAdvantageResult:
    """Advantages and the group statistics used to construct them."""

    advantages: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    normalizer: torch.Tensor


@dataclass(frozen=True)
class PolicyGradientResult:
    """A scalar loss plus diagnostics for one policy-gradient minibatch."""

    loss: torch.Tensor
    per_token_loss: torch.Tensor
    importance_weight: torch.Tensor
    clip_fraction: torch.Tensor
    effective_sequences: torch.Tensor


@dataclass(frozen=True)
class PPOTokenSurrogate:
    """Token-wise PPO terms and the resulting score-gradient coefficients.

    In an autoregressive language model, each response token is one action.
    ``gradient_coefficient`` is the scalar multiplying
    ``grad log pi_theta(token | prefix)`` under gradient ascent.  It is
    ``ratio * advantage`` while the pessimistic surrogate is live and exactly
    zero in the signed saturation region.
    """

    ratio: torch.Tensor
    clipped_ratio: torch.Tensor
    surrogate: torch.Tensor
    clipping_active: torch.Tensor
    gradient_coefficient: torch.Tensor


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Mean over selected entries, with an explicit all-masked convention of zero."""

    if values.shape != mask.shape:
        raise ValueError("values and mask must have matching shapes")
    weights = mask.to(dtype=values.dtype)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim)
    return numerator / denominator.clamp_min(1.0)


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    group_size: int | None = None,
    baseline: Baseline = "mean",
    normalizer: AdvantageNormalizer = "std",
    epsilon: float = 1e-6,
    unbiased_std: bool = False,
) -> GroupAdvantageResult:
    """Apply a prompt-group baseline and optional difficulty reweighting.

    ``rewards`` may have shape ``(batch, group)`` or be flat with an explicit
    ``group_size``.  Population standard deviation is the default so a binary
    group ``[0, 1]`` maps to ``[-1, 1]``; set ``unbiased_std=True`` to reproduce
    PyTorch's sample-standard-deviation convention.
    """

    if not rewards.is_floating_point():
        rewards = rewards.float()
    if rewards.ndim == 1:
        if group_size is None or group_size < 2 or rewards.numel() % group_size:
            raise ValueError("flat rewards require a group_size >= 2 that divides the batch")
        grouped = rewards.reshape(-1, group_size)
    elif rewards.ndim >= 2:
        if group_size is not None and rewards.shape[-1] != group_size:
            raise ValueError("group_size must match the final reward dimension")
        if rewards.shape[-1] < 2:
            raise ValueError("each prompt group needs at least two responses")
        grouped = rewards
    else:  # pragma: no cover - tensors are never negative-dimensional
        raise ValueError("rewards must have at least one dimension")
    if baseline not in ("mean", "none"):
        raise ValueError("baseline must be 'mean' or 'none'")
    if normalizer not in ("std", "mean", "none"):
        raise ValueError("normalizer must be 'std', 'mean', or 'none'")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    group_mean = grouped.mean(dim=-1, keepdim=True)
    group_std = grouped.std(dim=-1, keepdim=True, unbiased=unbiased_std)
    centered = grouped - group_mean if baseline == "mean" else grouped
    if normalizer == "std":
        scale = group_std
    elif normalizer == "mean":
        # MaxRL is defined for nonnegative verifier rewards.  Clamping rather
        # than taking abs keeps the implementation honest about that domain.
        if bool((group_mean < 0).any()):
            raise ValueError("mean normalization requires nonnegative group means")
        scale = group_mean
    else:
        scale = torch.ones_like(group_mean)
    safe_scale = scale.clamp_min(epsilon)
    advantages = centered / safe_scale
    # Equal-reward, mean-baseline groups have no relative information.  Force
    # exact zeros so epsilon cannot turn numerical noise into a training signal.
    if baseline == "mean":
        advantages = torch.where(group_std > epsilon, advantages, torch.zeros_like(advantages))
    return GroupAdvantageResult(
        advantages=advantages.reshape_as(rewards),
        group_mean=group_mean,
        group_std=group_std,
        normalizer=scale,
    )


def aggregate_token_losses(
    per_token_loss: torch.Tensor,
    mask: torch.Tensor,
    *,
    normalization: LossNormalization = "sequence",
    constant: float | None = None,
) -> torch.Tensor:
    """Reduce token losses using the denominator named in the objective."""

    if per_token_loss.ndim != 2 or mask.shape != per_token_loss.shape:
        raise ValueError("per_token_loss and mask must have shape batch x tokens")
    valid = mask.to(dtype=per_token_loss.dtype)
    if normalization == "sequence":
        return masked_mean(per_token_loss, mask, dim=-1).mean()
    if normalization == "token":
        return masked_mean(per_token_loss, mask)
    if normalization == "constant":
        if constant is None or constant <= 0:
            raise ValueError("positive constant is required for constant normalization")
        return (per_token_loss * valid).sum() / constant
    raise ValueError("normalization must be 'sequence', 'token', or 'constant'")


def ppo_token_surrogate(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> PPOTokenSurrogate:
    """Expose PPO as token-level importance reweighting of a score gradient.

    The samples come from ``pi_old``.  Their selected-action probabilities are
    reweighted by ``ratio = pi_theta / pi_old`` while the state/prefix
    distribution remains the one induced by ``pi_old``.  PPO therefore uses a
    near-on-policy, deliberately biased surrogate rather than an exact
    correction to the full new-policy trajectory distribution.

    Positive advantages saturate only when ``ratio > 1 + epsilon``.  Negative
    advantages saturate only when ``ratio < 1 - epsilon``.  Merely lying outside
    the interval is not enough: the sign decides whether clipping is active.
    """

    if not (
        new_log_probabilities.shape
        == old_log_probabilities.shape
        == advantages.shape
    ):
        raise ValueError("new log probabilities, old log probabilities, and advantages must align")
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie in (0, 1)")

    detached_old = old_log_probabilities.detach()
    detached_advantages = advantages.detach().to(new_log_probabilities.dtype)
    ratio = torch.exp(new_log_probabilities - detached_old)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(
        ratio * detached_advantages,
        clipped_ratio * detached_advantages,
    )
    clipping_active = (
        ((detached_advantages > 0) & (ratio > 1.0 + clip_epsilon))
        | ((detached_advantages < 0) & (ratio < 1.0 - clip_epsilon))
    )
    gradient_coefficient = torch.where(
        clipping_active,
        torch.zeros_like(ratio),
        ratio * detached_advantages,
    )
    return PPOTokenSurrogate(
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        surrogate=surrogate,
        clipping_active=clipping_active,
        gradient_coefficient=gradient_coefficient,
    )


def policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    old_log_probs: torch.Tensor | None = None,
    importance_method: ImportanceMethod = "none",
    clip_epsilon: float | None = None,
    normalization: LossNormalization = "sequence",
    normalization_constant: float | None = None,
) -> PolicyGradientResult:
    """Compute on-policy or stale-rollout score-function objectives.

    ``none`` is the direct REINFORCE objective ``-A log pi``. ``noclip`` uses
    token importance ratios, ``grpo`` applies PPO's pessimistic token clip,
    ``gspo`` applies the clipped geometric-mean sequence ratio, and ``cispo``
    upper-clips the token ratio while retaining gradient through the ratio.
    """

    if policy_log_probs.ndim != 2 or response_mask.shape != policy_log_probs.shape:
        raise ValueError("log probabilities and response_mask must have shape batch x tokens")
    if advantages.ndim == 2 and advantages.shape[-1] == 1:
        advantages = advantages[:, 0]
    if advantages.ndim != 1 or advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("advantages must contain one scalar per sequence")
    if importance_method not in ("none", "noclip", "grpo", "gspo", "cispo"):
        raise ValueError("unknown importance_method")
    if importance_method != "none":
        if old_log_probs is None or old_log_probs.shape != policy_log_probs.shape:
            raise ValueError("off-policy objectives require aligned old_log_probs")
    if importance_method in ("grpo", "gspo", "cispo"):
        if clip_epsilon is None or not 0 < clip_epsilon < 1:
            raise ValueError("clipped objectives require clip_epsilon in (0, 1)")

    valid = response_mask.bool()
    advantage = advantages.detach().to(policy_log_probs.dtype)[:, None]
    zeros = torch.zeros_like(policy_log_probs)
    if importance_method == "none":
        importance_weight = torch.ones_like(policy_log_probs)
        per_token = -advantage * policy_log_probs
        clipped = zeros.bool()
    else:
        assert old_log_probs is not None
        log_ratio = policy_log_probs - old_log_probs.detach()
        token_ratio = torch.exp(log_ratio)
        if importance_method == "noclip":
            importance_weight = token_ratio
            per_token = -(token_ratio * advantage)
            clipped = zeros.bool()
        elif importance_method == "grpo":
            assert clip_epsilon is not None
            token_terms = ppo_token_surrogate(
                policy_log_probs,
                old_log_probs,
                advantage.expand_as(policy_log_probs),
                clip_epsilon=clip_epsilon,
            )
            importance_weight = token_ratio
            per_token = -token_terms.surrogate
            clipped = (token_ratio - 1).abs() > clip_epsilon
        elif importance_method == "cispo":
            assert clip_epsilon is not None
            # ``minimum`` preserves a gradient through the selected token ratio
            # until the upper cap is reached; negative ratios are not clipped.
            capped_ratio = torch.minimum(token_ratio, torch.full_like(token_ratio, 1 + clip_epsilon))
            importance_weight = token_ratio
            per_token = -(capped_ratio * advantage)
            clipped = token_ratio > 1 + clip_epsilon
        else:
            assert importance_method == "gspo" and clip_epsilon is not None
            sequence_log_ratio = masked_mean(log_ratio, valid, dim=-1)
            sequence_ratio = torch.exp(sequence_log_ratio)
            clipped_ratio = sequence_ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            sequence_advantage = advantage[:, 0]
            sequence_surrogate = torch.minimum(
                sequence_ratio * sequence_advantage,
                clipped_ratio * sequence_advantage,
            )
            # Repeating a sequence scalar over valid tokens lets the standard
            # sequence reducer recover exactly one term per completion.
            per_token = -sequence_surrogate[:, None].expand_as(policy_log_probs)
            importance_weight = sequence_ratio[:, None].expand_as(policy_log_probs)
            clipped = ((sequence_ratio - 1).abs() > clip_epsilon)[:, None].expand_as(valid)

    loss = aggregate_token_losses(
        per_token,
        valid,
        normalization=normalization,
        constant=normalization_constant,
    )
    clip_fraction = masked_mean(clipped.to(policy_log_probs.dtype), valid)
    effective_sequences = valid.any(dim=-1).to(policy_log_probs.dtype).sum()
    return PolicyGradientResult(
        loss=loss,
        per_token_loss=per_token,
        importance_weight=importance_weight,
        clip_fraction=clip_fraction,
        effective_sequences=effective_sequences,
    )


def reference_kl_estimate(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    normalization: LossNormalization = "sequence",
    normalization_constant: float | None = None,
) -> torch.Tensor:
    """Nonnegative Schulman ``k3`` estimate of KL(policy || reference)."""

    if policy_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy and reference log probabilities must align")
    log_reference_over_policy = reference_log_probs.detach() - policy_log_probs
    per_token = torch.exp(log_reference_over_policy) - log_reference_over_policy - 1.0
    return aggregate_token_losses(
        per_token,
        response_mask,
        normalization=normalization,
        constant=normalization_constant,
    )
