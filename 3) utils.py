# 3) utils.py
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from typing import List, Dict, Tuple

import numpy as np
import torch

SPECIAL_TOKENS = {
    "pad": "<PAD>",
    "unk": "<UNK>",
    "sos": "<SOS>",
    "eos": "<EOS>",
}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def simple_tokenize(text: str) -> List[str]:
    # Simple tokenizer: lowercase + keep words/numbers
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)

class Vocabulary:
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def build(self, captions: List[str]) -> None:
        counter = Counter()
        for cap in captions:
            counter.update(simple_tokenize(cap))

        words = [w for w, c in counter.items() if c >= self.min_freq]

        # Special tokens first
        vocab_list = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["unk"],
            SPECIAL_TOKENS["sos"],
            SPECIAL_TOKENS["eos"],
        ] + sorted(words)

        self.stoi = {w: i for i, w in enumerate(vocab_list)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = [SPECIAL_TOKENS["sos"]] + simple_tokenize(text) + [SPECIAL_TOKENS["eos"]]
        ids = [self.stoi.get(t, self.stoi[SPECIAL_TOKENS["unk"]]) for t in tokens]
        ids = ids[:max_len]
        # pad
        pad_id = self.stoi[SPECIAL_TOKENS["pad"]]
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            w = self.itos.get(int(i), SPECIAL_TOKENS["unk"])
            if w == SPECIAL_TOKENS["eos"]:
                break
            if w in (SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["sos"]):
                continue
            words.append(w)
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"min_freq": self.min_freq, "stoi": self.stoi}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = Vocabulary(min_freq=int(obj["min_freq"]))
        v.stoi = {k: int(vv) for k, vv in obj["stoi"].items()}
        v.itos = {i: w for w, i in v.stoi.items()}
        return v

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    return images, captions

def save_checkpoint(path: str, model, optimizer, epoch: int, best_score: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
        },
        path,
    )

def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    return ckpt
