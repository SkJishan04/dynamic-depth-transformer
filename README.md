![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-pytest-yellow.svg)

# Dynamic Depth Transformer: Latency-Aware Soft-Routing for Early Exit

Standard Transformers spend the same compute on "the" as on the hardest token in a mathematical proof. This project implements a **token-level early-exit Transformer** that learns *how deep* each token needs to go through the network, trained end-to-end with a **Gumbel-Softmax router** and a **hardware latency-aware joint loss** — no hand-crafted heuristics, no separate distillation stage, just a differentiable routing signal baked directly into training.

---

## Table of Contents

- [Core Idea](#core-idea)
- [Joint Objective](#joint-objective)
- [Results](#results-ag-news-24-layer-model-d_model256-trained-8-epochs)
- [Reasoning & Design Decisions](#reasoning--design-decisions)
- [Limitations & Future Work](#limitations--future-work)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Implementation Notes](#implementation-notes)
- [Tech Stack](#tech-stack)

---

## Core Idea

A lightweight binary router is placed after every 4th Transformer block. At each checkpoint, the router predicts `P(exit)` for every token independently. During training, a **Straight-Through Gumbel-Softmax estimator** makes this discrete exit/continue decision differentiable, so gradients from the downstream classification loss can flow back and shape routing behavior. At inference, tokens exit deterministically once `P(exit)` crosses a threshold **τ**, freezing their representation and skipping all remaining layers — no wasted compute on tokens the model is already confident about.

Token x ──► [Layer 1] ──► [Router 1] ──► (Exit Decision?) ──► Final Classification
│ (No)
▼
[Layer 2] ──► [Router 2] ──► ...


Each token in a sequence is routed **independently** — a short, common word can exit after 4 layers while a named entity or ambiguous term in the same sentence continues to layer 24. This is the key departure from sequence-level early exit (where the whole input either exits or doesn't): it lets the model allocate compute at the granularity where difficulty actually varies.

## Joint Objective

L_total = L_CE + γ · Σ_{i=1}^{L} (i · P_exit,i)


The cross-entropy term drives classification accuracy as usual. The latency term directly charges the model a cost proportional to *how deep* it sends each token — exiting at layer 20 costs 5x more penalty than exiting at layer 4. This means the model isn't just learning to classify; it's learning a **cost-aware policy** that trades off compute against confidence, entirely through gradient descent on a single unified loss.

---

## Results (AG News, 24-layer model, d_model=256, trained 8 epochs)

At the optimal operating point (**τ = 0.2**):

| Metric | Value |
|---|---|
| Accuracy | **90.79%** |
| Avg. layers traversed (all tokens) | 8.57 / 24 |
| Avg. layers — easy tokens (bottom 25%) | **4.00** |
| Avg. layers — hard tokens (top 25%) | **22.46** |
| FLOPs saved vs. dense baseline | **64.31%** |

![Pareto Frontier](results/pareto_frontier.png)