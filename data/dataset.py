"""
AG News dataset loader with a simple from-scratch vocabulary.

AG News (4-class topic classification: World / Sports / Business / Sci-Tech)
is a good testbed for early exit: many headlines are short and lexically
simple ("Stocks rise today") while others require multi-hop reasoning about
named entities and numbers, giving genuine easy/hard token variation to
showcase in the results.
"""

import re
from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

PAD, UNK = "<pad>", "<unk>"


def simple_tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def build_vocab(texts, vocab_size=30000):
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    most_common = counter.most_common(vocab_size - 2)
    vocab = {PAD: 0, UNK: 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab


def encode(text, vocab, max_len):
    tokens = simple_tokenize(text)[:max_len]
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokens]
    attention_mask = [1] * len(ids)
    pad_len = max_len - len(ids)
    ids += [vocab[PAD]] * pad_len
    attention_mask += [0] * pad_len
    return ids, attention_mask


class AGNewsDataset(Dataset):
    def __init__(self, split, vocab, max_len=128):
        raw = load_dataset("ag_news", split=split)
        self.texts = raw["text"]
        self.labels = raw["label"]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, mask = encode(self.texts[idx], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_ag_news(vocab_size=30000, max_len=128):
    train_raw = load_dataset("ag_news", split="train")
    vocab = build_vocab(train_raw["text"], vocab_size=vocab_size)

    train_ds = AGNewsDataset("train", vocab, max_len)
    test_ds = AGNewsDataset("test", vocab, max_len)
    return train_ds, test_ds, vocab