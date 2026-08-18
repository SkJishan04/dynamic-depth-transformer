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

