from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # --------- Paths (change to your local paths) ----------
    data_dir: str = "./data"
    train_img_dir: str = "./data/train2017"
    val_img_dir: str = "./data/val2017"
    train_ann: str = "./data/annotations/captions_train2017.json"
    val_ann: str = "./data/annotations/captions_val2017.json"

    # --------- Results / checkpoints ----------
    out_dir: str = "./results"
    ckpt_path: str = "./results/best.pt"
    vocab_path: str = "./results/vocab.json"

    # --------- Training ----------
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-3
    max_caption_len: int = 30
    min_word_freq: int = 5

    # --------- Model ----------
    embed_size: int = 256
    hidden_size: int = 512
    num_layers: int = 1
    dropout: float = 0.3
    freeze_cnn: bool = True  # set False to fine-tune CNN

    # --------- Device ----------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# Ensure output directory exists
Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)
