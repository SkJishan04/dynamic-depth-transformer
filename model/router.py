"""
Gumbel-Softmax binary router for token-level early exit.

At each router checkpoint (placed after every `router_every` blocks), each
token's hidden state is fed through a small MLP producing 2 logits:
[continue, exit]. During training we use the Straight-Through Gumbel-Softmax
trick so the discrete exit decision is differentiable. During inference we
threshold P(exit) against tau for a hard, deterministic skip decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRouter(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # logits: [continue, exit]
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0,
                hard: bool = False, tau: float = None):
        """
        Args:
            x: (batch, seq_len, d_model) hidden states
            temperature: Gumbel-Softmax temperature (annealed during training)
            hard: if True, use straight-through hard sampling (training) or
                  threshold-based hard decision (inference, requires tau)
            tau: exit probability threshold, used only at inference

        Returns:
            exit_prob: (batch, seq_len) soft P(exit), used for latency loss
            exit_mask: (batch, seq_len) binary {0,1} decision mask
        """
        logits = self.net(x)  # (B, S, 2)

        if self.training:
            # Straight-through Gumbel-Softmax: soft gradients, hard forward pass
            gumbel_out = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            exit_prob_soft = F.softmax(logits, dim=-1)[..., 1]
            exit_mask = gumbel_out[..., 1]
            return exit_prob_soft, exit_mask
        else:
            # Deterministic threshold-based decision at inference
            exit_prob = F.softmax(logits, dim=-1)[..., 1]
            assert tau is not None, "tau must be provided at inference time"
            exit_mask = (exit_prob > tau).float()
            return exit_prob, exit_mask


def anneal_temperature(step: int, temp_start: float, temp_min: float,
                        anneal_steps: int) -> float:
    """Linearly anneal Gumbel temperature from temp_start down to temp_min."""
    frac = min(step / max(anneal_steps, 1), 1.0)
    return temp_start - frac * (temp_start - temp_min)