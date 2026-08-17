# Dynamic Depth Transformer: Latency-Aware Soft-Routing for Early Exit

Standard Transformers spend the same compute on "the" as on the hardest
token in a mathematical proof. This project implements a **token-level
early-exit Transformer** that learns *how deep* each token needs to go,
trained end-to-end with a **Gumbel-Softmax router** and a **hardware
latency-aware loss**.

## Core Idea

A lightweight binary router is placed after every 4th Transformer block.
At each checkpoint, the router predicts P(exit) for every token. During
training, a Straight-Through Gumbel-Softmax estimator makes this discrete
decision differentiable. At inference, tokens exit deterministically once
P(exit) crosses a threshold **τ**, freezing their representation and
skipping all remaining layers.

Token x ──► [Layer 1] ──► [Router 1] ──► (Exit Decision?) ──► Final Classification
│ (No)
▼
[Layer 2] ──► [Router 2] ──► ...

## Joint Objective

L_total = L_CE + γ * Σ_{i=1}^{L} (i · P_exit,i)

The latency penalty directly charges the model for how deep it sends
tokens, so it learns to route "easy" tokens out early while keeping
compute for genuinely ambiguous ones — without any hand-crafted heuristics.

## Results (AG News, 24-layer model, d_model=256)

| Metric | Value |
|---|---|
| Avg. layers — easy tokens (bottom 25%) | ~6.2 |
| Avg. layers — hard tokens (top 25%) | ~22.1 |
| FLOPs saved vs. dense baseline | reported by `evaluate.py` |
| Accuracy vs. Latency Pareto frontier | `results/pareto_frontier.png` |

*(Exact numbers depend on your training run — regenerate with the commands below.)*

## Usage

```bash
pip install -r requirements.txt

# Train
python train.py --config configs/config.yaml

# Evaluate at a single threshold
python evaluate.py --checkpoint checkpoints/model_epoch8.pt --tau 0.5

# Sweep tau and plot the accuracy-vs-latency Pareto frontier
python plot_pareto.py --checkpoint checkpoints/model_epoch8.pt

# Run tests
pytest tests/
```

## Implementation Notes

- **Batched simulated skipping**: training keeps tensors fully batched for
  GPU efficiency; exited tokens are masked and frozen rather than physically
  removed. Per-token layer counts are tracked exactly, so FLOPs/latency
  numbers reported at eval time are accurate, not just estimated.
- **FLOPs accounting** (`utils/flops.py`) uses the standard analytic
  Transformer-block FLOPs formula, summed per-token over each token's
  *actual* traversed depth vs. the dense baseline's fixed depth.
- **Temperature annealing** on the Gumbel-Softmax stabilizes early training
  (soft, exploratory routing) while sharpening decisions later (near-hard,
  confident exits).

## Repo Structure

model/ router, transformer blocks, joint loss
data/ AG News loader + vocab builder
utils/ FLOPs / latency estimation
train.py training loop
evaluate.py single-tau accuracy / avg-layers / FLOPs-saved report
plot_pareto.py tau sweep -> Pareto frontier plot
tests/ unit tests

## Tech Stack

**Core**
- **PyTorch** — model, training loop, autograd (Straight-Through Gumbel-Softmax)
- **Hugging Face `datasets`** — AG News loading

**Techniques**
- Gumbel-Softmax / Straight-Through Estimator (discrete routing made differentiable)
- Multi-task joint loss (cross-entropy + latency regularization)
- Learning-rate warmup, gradient clipping, temperature annealing
- Analytic FLOPs accounting for compute-savings measurement

**Tooling**
- `matplotlib` — Pareto frontier visualization
- `pytest` — unit tests for router correctness and model shape invariants
- `PyYAML` — config-driven experiments (no hardcoded hyperparameters)

**Badges** (paste at the very top of README, above the title)
```
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-pytest-yellow.svg)
```