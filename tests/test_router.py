"""
Unit tests for the Gumbel-Softmax router and end-to-end model shape checks.
Run with: pytest tests/
"""

import torch

from model.router import GumbelRouter, anneal_temperature
from model.transformer import DynamicDepthTransformer


def test_router_output_shapes():
    router = GumbelRouter(d_model=32)
    x = torch.randn(4, 10, 32)

    router.train()
    exit_prob, exit_mask = router(x, temperature=1.0, hard=True)
    assert exit_prob.shape == (4, 10)
    assert exit_mask.shape == (4, 10)
    assert torch.all((exit_mask == 0) | (exit_mask == 1))


def test_router_inference_threshold():
    router = GumbelRouter(d_model=32)
    router.eval()
    x = torch.randn(4, 10, 32)
    exit_prob, exit_mask = router(x, tau=0.5)
    assert torch.all((exit_mask == 0) | (exit_mask == 1))
    assert torch.all((exit_prob >= 0) & (exit_prob <= 1))


def test_temperature_annealing_bounds():
    t0 = anneal_temperature(0, temp_start=1.0, temp_min=0.3, anneal_steps=1000)
    t_mid = anneal_temperature(500, temp_start=1.0, temp_min=0.3, anneal_steps=1000)
    t_end = anneal_temperature(10000, temp_start=1.0, temp_min=0.3, anneal_steps=1000)
    assert t0 == 1.0
    assert 0.3 < t_mid < 1.0
    # assert t_end == 0.3
    assert abs(t_end - 0.3) < 1e-9


def test_model_forward_shapes_and_valid_layer_counts():
    model = DynamicDepthTransformer(
        vocab_size=100, d_model=32, n_heads=4, d_ff=64,
        n_layers=8, router_every=4, num_classes=4, max_seq_len=16,
    )
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    attention_mask[0, 10:] = 0  # simulate padding on example 0

    model.eval()
    logits, info = model(input_ids, attention_mask, tau=0.5)

    assert logits.shape == (2, 4)
    layers_traversed = info["layers_traversed"]
    assert layers_traversed.shape == (2, 16)
    # No real token should traverse more than n_layers, no token fewer than 0
    assert torch.all(layers_traversed <= 8)
    assert torch.all(layers_traversed >= 0)
    # Padding positions should not accumulate layer counts
    assert torch.all(layers_traversed[0, 10:] == 0)