"""
State discretizer for converting continuous IsaacLab CartPole observations
into discrete bin indices suitable for tabular Q-learning.

Observation format from IsaacLab (obs["policy"]):
    Tensor of shape (num_envs, 4):
        [cart_position, pole_angle, cart_velocity, pole_angular_velocity]

Each continuous dimension is clipped to its bounds, then mapped to one of
N bins using uniform spacing. The result is a tuple of integers that serves
as a dictionary key for the Q-table.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple, Union


class StateDiscretizer:
    """
    Discretize a continuous 4D CartPole observation into bin indices.

    Args:
        bins: Number of bins per observation dimension, e.g. [8, 16, 8, 8].
        obs_bounds: (low, high) clipping bounds per dimension.

    Usage:
        discretizer = StateDiscretizer(bins=[8,16,8,8], obs_bounds=[...])
        state_key = discretizer.discretize(obs)   # returns (2, 7, 4, 3)
    """

    def __init__(
        self,
        bins: List[int],
        obs_bounds: List[Tuple[float, float]],
    ) -> None:
        assert len(bins) == len(obs_bounds), "bins and obs_bounds must have same length"
        self.n_dims = len(bins)
        self.bins = bins
        self.obs_bounds = obs_bounds

        # Pre-compute bin edges for each dimension (np.linspace creates
        # bins+1 edges, digitize maps values into 1..bins)
        self.bin_edges = []
        for i in range(self.n_dims):
            low, high = obs_bounds[i]
            # bins[i]+1 edges → bins[i] intervals
            edges = np.linspace(low, high, bins[i] + 1)
            self.bin_edges.append(edges)

    def discretize(self, obs: Union[dict, torch.Tensor, np.ndarray]) -> tuple:
        """
        Convert an observation into a tuple of bin indices.

        Args:
            obs: One of:
                - dict with key "policy" → Tensor of shape (1, 4) or (4,)
                - Tensor of shape (1, 4) or (4,)
                - ndarray of shape (4,)

        Returns:
            Tuple[int, ...]: Bin indices, e.g. (2, 7, 4, 3).
        """
        # ---- Extract raw numpy array from IsaacLab observation ---- #
        if isinstance(obs, dict):
            raw = obs.get("policy", next(iter(obs.values())))
        else:
            raw = obs

        if isinstance(raw, torch.Tensor):
            arr = raw.detach().cpu().numpy().flatten()
        else:
            arr = np.asarray(raw).flatten()

        assert len(arr) >= self.n_dims, (
            f"Expected at least {self.n_dims} obs values, got {len(arr)}"
        )

        # ---- Clip and digitize each dimension ---- #
        indices = []
        for i in range(self.n_dims):
            low, high = self.obs_bounds[i]
            val = np.clip(arr[i], low, high)
            # np.digitize returns 0..bins[i]; we clip to [0, bins[i]-1]
            bin_idx = int(np.digitize(val, self.bin_edges[i])) - 1
            bin_idx = max(0, min(bin_idx, self.bins[i] - 1))
            indices.append(bin_idx)

        return tuple(indices)

    @property
    def total_states(self) -> int:
        """Total number of discrete states (product of all bin counts)."""
        result = 1
        for b in self.bins:
            result *= b
        return result

    def __repr__(self) -> str:
        return (
            f"StateDiscretizer(bins={self.bins}, "
            f"obs_bounds={self.obs_bounds}, "
            f"total_states={self.total_states})"
        )
