"""
Custom feature extractor that is aware of the MPC structure.
The residual policy observes robot state, recent wheel commands, stuck
indicators, current u_mpc, and stuck-gated control authority. It outputs
normalized residual corrections that the environment scales before combining
with the MPC action.

This is just a thin wrapper around SB3's MlpPolicy, but having it as a
separate file makes the architecture clearer and lets you swap networks easily.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np


class ResidualFeaturesExtractor(BaseFeaturesExtractor):
    """
    Slightly deeper MLP with layer norm for the 29-dim climbing observation.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# Policy kwargs to pass to SB3 PPO/SAC
RESIDUAL_POLICY_KWARGS = dict(
    features_extractor_class=ResidualFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[128, 64], vf=[128, 64]),
    activation_fn=nn.Tanh,
)

ARM_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    activation_fn=nn.ReLU,
)
