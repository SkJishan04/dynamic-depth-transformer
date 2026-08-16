"""
Joint loss: cross-entropy + hardware-latency penalty.

    L_total = L_CE + gamma * sum_i (i * P_exit_i)

where i is the checkpoint's layer index and P_exit_i is the router's soft
exit probability at that checkpoint, averaged over real (non-padding) tokens
in the batch. This directly penalizes the model for pushing tokens deeper
into the network, encouraging it to learn which tokens can exit early
without hurting accuracy.
"""

import torch
import torch.nn as nn


class DynamicDepthLoss(nn.Module):
    def __init__(self, gamma: float = 0.01):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, info):
        ce_loss = self.ce(logits, labels)

        attention_mask = info["attention_mask"].float()
        n_real_tokens = attention_mask.sum().clamp(min=1.0)

        latency_penalty = 0.0
        for layer_idx, exit_prob in info["exit_probs"]:
            # Only count penalty over real tokens
            masked_prob = exit_prob * attention_mask
            latency_penalty = latency_penalty + layer_idx * masked_prob.sum()

        latency_penalty = latency_penalty / n_real_tokens
        total_loss = ce_loss + self.gamma * latency_penalty

        return total_loss, {
            "ce_loss": ce_loss.item(),
            "latency_penalty": latency_penalty.item() if torch.is_tensor(latency_penalty) else latency_penalty,
            "total_loss": total_loss.item(),
        }