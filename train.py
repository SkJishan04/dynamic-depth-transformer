"""
Training script for the Dynamic-Depth Transformer.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import load_ag_news
from model.transformer import DynamicDepthTransformer
from model.router import anneal_temperature
from model.losses import DynamicDepthLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading AG News + building vocab...")
    train_ds, test_ds, vocab = load_ag_news(
        vocab_size=cfg["model"]["vocab_size"],
        max_len=cfg["model"]["max_seq_len"],
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                               shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=2)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    loss_fn = DynamicDepthLoss(gamma=cfg["training"]["gamma"])

    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    global_step = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            temperature = anneal_temperature(
                global_step,
                cfg["model"]["gumbel_temp"],
                cfg["model"]["gumbel_temp_min"],
                cfg["model"]["gumbel_temp_anneal_steps"],
            )

            logits, info = model(input_ids, attention_mask, temperature=temperature)
            loss, log_dict = loss_fn(logits, labels, info)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=log_dict["total_loss"], ce=log_dict["ce_loss"],
                              lat=log_dict["latency_penalty"], temp=round(temperature, 3))

        # quick eval each epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits, _ = model(input_ids, attention_mask,
                                   tau=cfg["inference"]["exit_threshold"])
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch+1}: test accuracy = {correct/total:.4f}")

        ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], f"model_epoch{epoch+1}.pt")
        torch.save({"model_state": model.state_dict(), "vocab": vocab, "config": cfg}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()