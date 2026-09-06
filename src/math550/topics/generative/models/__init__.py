"""Public pedagogical generative-model implementations."""

from .diffusion import DenoisingDiffusion, linear_beta_schedule, make_diffusion_training_data
from .flow_matching import ConditionalFlowMatcher, make_flow_matching_training_data
from .image_flow import ConditionalMNISTFlowMatcher, MNISTVelocityCNN
from .neural import (
    SUPPORTED_TORCH_DEVICES,
    TimeConditionedMLP,
    TorchVectorRegressor,
    resolve_torch_device,
)
from .normalizing_flow import AffineCouplingLayer, GaussianAffineFlow
from .tutorial_flow import TutorialVelocityMLP, TwoMoonsFlowMatcher
from .vae import (
    ConditionalMNISTVAE,
    ConditionalMNISTVAEModel,
    GaussianVAE,
    VectorVAE,
    diagonal_gaussian_kl,
)

__all__ = [
    "AffineCouplingLayer",
    "ConditionalFlowMatcher",
    "ConditionalMNISTFlowMatcher",
    "ConditionalMNISTVAE",
    "ConditionalMNISTVAEModel",
    "DenoisingDiffusion",
    "GaussianAffineFlow",
    "GaussianVAE",
    "MNISTVelocityCNN",
    "SUPPORTED_TORCH_DEVICES",
    "TimeConditionedMLP",
    "TorchVectorRegressor",
    "TutorialVelocityMLP",
    "TwoMoonsFlowMatcher",
    "VectorVAE",
    "diagonal_gaussian_kl",
    "linear_beta_schedule",
    "make_diffusion_training_data",
    "make_flow_matching_training_data",
    "resolve_torch_device",
]
