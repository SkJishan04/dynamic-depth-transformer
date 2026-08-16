"""
Evaluation script: measures accuracy, average layers traversed per token
(overall + easy vs. hard token breakdown), and FLOPs saved relative to the
dense (all-layers) baseline, at a single exit threshold tau.

Usage:
    python evaluate.py --checkpoint checkpoints/model_epoch8.pt --tau 0.5
"""

import argparse

import torch
from torch.utils.data import DataLoader

from data.dataset import load_ag_news, AGNewsDataset
from model.transformer import DynamicDepthTransformer
from utils.flops import FlopsConfig, total_flops_baseline, total_flops_early_exit, flops_saved_pct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    vocab = ckpt["vocab"]

    model = DynamicDepthTransformer(
        vocab_size=len(vocab),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"],
        router_every=cfg["model"]["router_every"],
        num_classes=cfg["training"]["num_classes"],
        max_seq_len=cfg["model"]["max_seq_len"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, vocab, cfg


@torch.no_grad()
def evaluate(model, loader, tau, device, cfg):
    model.eval()
    correct, total = 0, 0

    all_layers_per_token = []          # flat list across dataset (real tokens only)
    per_example_avg_layers = []        # avg layers per example -> used for easy/hard split

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits, info = model(input_ids, attention_mask, tau=tau)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        layers_traversed = info["layers_traversed"]  # (B, S)
        mask = attention_mask.bool()

        real_layers = layers_traversed[mask].tolist()
        all_layers_per_token.extend(real_layers)

        # per-example mean (for easy/hard breakdown by example difficulty)
        token_counts = attention_mask.sum(dim=1).clamp(min=1)
        example_means = (layers_traversed * attention_mask).sum(dim=1) / token_counts
        per_example_avg_layers.extend(example_means.tolist())

    accuracy = correct / total
    avg_layers = sum(all_layers_per_token) / len(all_layers_per_token)

    # "Easy" = bottom 25% of examples by avg layers traversed, "hard" = top 25%
    sorted_avgs = sorted(per_example_avg_layers)
    n = len(sorted_avgs)
    easy_avg = sum(sorted_avgs[: n // 4]) / max(n // 4, 1)
    hard_avg = sum(sorted_avgs[-(n // 4):]) / max(n // 4, 1)

    n_layers = cfg["model"]["n_layers"]
    flops_cfg = FlopsConfig(
        d_model=cfg["model"]["d_model"],
        d_ff=cfg["model"]["d_ff"],
        n_heads=cfg["model"]["n_heads"],
        seq_len=cfg["model"]["max_seq_len"],
    )
    baseline = total_flops_baseline(flops_cfg, n_layers, len(all_layers_per_token))
    actual = total_flops_early_exit(flops_cfg, all_layers_per_token)
    saved_pct = flops_saved_pct(baseline, actual)

    return {
        "tau": tau,
        "accuracy": accuracy,
        "avg_layers_traversed": avg_layers,
        "easy_tokens_avg_layers": easy_avg,
        "hard_tokens_avg_layers": hard_avg,
        "flops_baseline": baseline,
        "flops_actual": actual,
        "flops_saved_pct": saved_pct,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab, cfg = load_model(args.checkpoint, device)
    _, test_ds, _ = load_ag_news(vocab_size=cfg["model"]["vocab_size"],
                                  max_len=cfg["model"]["max_seq_len"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    results = evaluate(model, test_loader, args.tau, device, cfg)

    print("\n===== Dynamic Depth Evaluation =====")
    print(f"tau (exit threshold):        {results['tau']}")
    print(f"Accuracy:                    {results['accuracy']:.4f}")
    print(f"Avg layers traversed:        {results['avg_layers_traversed']:.2f}")
    print(f"  Easy tokens (bottom 25%):  {results['easy_tokens_avg_layers']:.2f} layers")
    print(f"  Hard tokens (top 25%):     {results['hard_tokens_avg_layers']:.2f} layers")
    print(f"FLOPs saved vs. dense model: {results['flops_saved_pct']:.2f}%")


if __name__ == "__main__":
    main()