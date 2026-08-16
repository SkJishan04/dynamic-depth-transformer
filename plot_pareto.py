"""
Sweeps the exit threshold tau and plots the Accuracy vs. Average Latency
Pareto frontier — the headline result for this project.

Usage:
    python plot_pareto.py --checkpoint checkpoints/model_epoch8.pt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import load_ag_news
from evaluate import evaluate, load_model
from utils.flops import FlopsConfig, estimate_latency_ms, flops_per_token_per_layer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", type=str, default="results/pareto_frontier.png")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, model_cfg = load_model(args.checkpoint, device)
    _, test_ds, _ = load_ag_news(vocab_size=model_cfg["model"]["vocab_size"],
                                  max_len=model_cfg["model"]["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    flops_cfg = FlopsConfig(
        d_model=model_cfg["model"]["d_model"],
        d_ff=model_cfg["model"]["d_ff"],
        n_heads=model_cfg["model"]["n_heads"],
        seq_len=model_cfg["model"]["max_seq_len"],
    )
    per_layer_flops = flops_per_token_per_layer(flops_cfg)

    all_results = []
    for tau in cfg["inference"]["tau_sweep"]:
        res = evaluate(model, test_loader, tau, device, model_cfg)
        avg_token_flops = per_layer_flops * res["avg_layers_traversed"]
        res["latency_ms"] = estimate_latency_ms(
            avg_token_flops * model_cfg["model"]["max_seq_len"]
        )
        all_results.append(res)
        print(f"tau={tau:.2f}  acc={res['accuracy']:.4f}  "
              f"avg_layers={res['avg_layers_traversed']:.2f}  "
              f"latency={res['latency_ms']:.3f}ms  "
              f"flops_saved={res['flops_saved_pct']:.1f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/pareto_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot
    latencies = [r["latency_ms"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]
    taus = [r["tau"] for r in all_results]

    plt.figure(figsize=(7, 5))
    plt.plot(latencies, accuracies, marker="o", linewidth=2)
    for x, y, t in zip(latencies, accuracies, taus):
        plt.annotate(f"τ={t}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Average Latency per Sequence (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Latency Pareto Frontier (Dynamic Depth Transformer)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved Pareto frontier plot to {args.out}")


if __name__ == "__main__":
    main()