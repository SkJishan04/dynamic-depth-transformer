"""
FLOPs estimation utilities for measuring compute savings from early exit.

We use the standard analytic approximation for a Transformer block's
forward-pass FLOPs per token (Kaplan et al. 2020 / Chinchilla-style counting):

    FLOPs_per_token_per_layer ≈ 2 * (4 * d_model^2 + 2 * d_model * d_ff)

This counts the QKVO projections (4 * d_model^2) and the two FFN matmuls
(2 * d_model * d_ff), each multiply-accumulate counted as 2 FLOPs.
Attention score compute (O(seq_len * d_model)) is comparatively small for
short sequences and is added as a correction term.
"""

from dataclasses import dataclass


@dataclass
class FlopsConfig:
    d_model: int
    d_ff: int
    n_heads: int
    seq_len: int


def flops_per_token_per_layer(cfg: FlopsConfig) -> float:
    """Approximate FLOPs to push a single token through a single Transformer block."""
    attn_proj_flops = 4 * cfg.d_model ** 2          # Q, K, V, O projections
    ffn_flops = 2 * cfg.d_model * cfg.d_ff           # up-proj + down-proj
    attn_score_flops = 2 * cfg.d_model * cfg.seq_len  # QK^T + softmax*V, amortized per token
    return 2 * (attn_proj_flops + ffn_flops) + attn_score_flops


def total_flops_baseline(cfg: FlopsConfig, n_layers: int, n_tokens: int) -> float:
    """FLOPs if every token traverses all n_layers (standard dense Transformer)."""
    return flops_per_token_per_layer(cfg) * n_layers * n_tokens


def total_flops_early_exit(cfg: FlopsConfig, layers_per_token: list) -> float:
    """
    FLOPs actually spent given a per-token list of how many layers each token
    traversed before exiting.
    """
    per_layer = flops_per_token_per_layer(cfg)
    return per_layer * sum(layers_per_token)


def flops_saved_pct(baseline_flops: float, actual_flops: float) -> float:
    if baseline_flops == 0:
        return 0.0
    return 100.0 * (1.0 - actual_flops / baseline_flops)


def estimate_latency_ms(flops: float, device_tflops: float = 20.0) -> float:
    """
    Convert FLOPs to a rough wall-clock latency estimate given an assumed
    device throughput (default: 20 TFLOPs/s, a conservative single-GPU
    effective throughput after accounting for memory-bound ops).
    """
    seconds = flops / (device_tflops * 1e12)
    return seconds * 1000.0