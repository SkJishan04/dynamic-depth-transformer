"""
Dynamic-Depth Transformer with per-token early exit.

Architecture:
    Token embeddings + positional encoding
    -> N TransformerBlocks
    -> a GumbelRouter after every `router_every` blocks
    -> tokens whose exit decision fires have their hidden state FROZEN
       (masked out of further updates) and their exit layer recorded
    -> final classification head pools over each token's frozen/final state

Because we train in batched (padded) mode, we simulate skipping via masking
rather than physically shortening tensors: an "active mask" tracks which
tokens are still being updated. This keeps training GPU-friendly while giving
an exact accounting of how many layers each token effectively traversed,
which is what we use for the latency loss and FLOPs analysis at eval time.
"""

import math
import torch
import torch.nn as nn

from model.router import GumbelRouter


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                                 need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class DynamicDepthTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 router_every, num_classes, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.router_every = router_every
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # One router per checkpoint (every `router_every` layers)
        self.checkpoint_layers = list(range(router_every, n_layers + 1, router_every))
        self.routers = nn.ModuleDict({
            str(layer_idx): GumbelRouter(d_model)
            for layer_idx in self.checkpoint_layers
        })

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask, temperature=1.0, tau=0.5):
        """
        Args:
            input_ids: (B, S)
            attention_mask: (B, S) 1 = real token, 0 = padding
            temperature: Gumbel-Softmax temperature (training only)
            tau: exit threshold (inference only)

        Returns:
            logits: (B, num_classes)
            info: dict with per-checkpoint exit_probs (for latency loss) and
                  per-token layer counts (for FLOPs/latency measurement)
        """
        B, S = input_ids.shape
        device = input_ids.device
        key_padding_mask = (attention_mask == 0)  # True = pad, ignored by attention

        x = self.dropout(self.pos_enc(self.embed(input_ids)))

        # active_mask: 1.0 = token still being updated, 0.0 = has exited
        active_mask = attention_mask.float().clone()  # padding starts "inactive"
        layers_traversed = torch.zeros(B, S, device=device)
        all_exit_probs = []  # for latency loss: list of (checkpoint_layer, exit_prob)

        final_hidden = torch.zeros(B, S, self.d_model, device=device)

        for layer_idx, block in enumerate(self.blocks, start=1):
            block_out = block(x, key_padding_mask=key_padding_mask)

            # Only tokens still active get updated; exited tokens are frozen
            active_expanded = active_mask.unsqueeze(-1)
            x = active_expanded * block_out + (1 - active_expanded) * x

            # Real (non-padding) tokens that are still active just traversed this layer
            counted = active_mask * attention_mask.float()
            layers_traversed += counted

            if layer_idx in self.checkpoint_layers:
                router = self.routers[str(layer_idx)]
                exit_prob, exit_decision = router(
                    x, temperature=temperature,
                    hard=self.training, tau=tau
                )
                all_exit_probs.append((layer_idx, exit_prob))

                # A token can only exit if it is (a) real, (b) currently active
                newly_exited = exit_decision * active_mask * attention_mask.float()

                # Snapshot hidden state for tokens exiting right now
                final_hidden = final_hidden + newly_exited.unsqueeze(-1) * x

                # Deactivate them for subsequent layers
                active_mask = active_mask * (1 - newly_exited)

        # Any tokens that never triggered exit ride to the final layer
        final_hidden = final_hidden + active_mask.unsqueeze(-1) * x * attention_mask.float().unsqueeze(-1)

        # Mean-pool over real tokens' final representations -> sequence classification
        mask_f = attention_mask.float().unsqueeze(-1)
        pooled = (final_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        logits = self.classifier(pooled)

        info = {
            "exit_probs": all_exit_probs,          # for latency loss
            "layers_traversed": layers_traversed,   # (B, S) for FLOPs/latency measurement
            "attention_mask": attention_mask,
        }
        return logits, info