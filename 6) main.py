import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG
from dataset import CocoCaptionDataset, load_all_captions
from model import EncoderCNN, DecoderRNN, CaptioningModel
from utils import seed_everything, Vocabulary, collate_fn, save_checkpoint, load_checkpoint, SPECIAL_TOKENS

def compute_bleu(model: CaptioningModel, loader: DataLoader, vocab: Vocabulary, device: str) -> tuple:
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    references = []
    hypotheses = []

    sos_id = vocab.stoi[SPECIAL_TOKENS["sos"]]
    eos_id = vocab.stoi[SPECIAL_TOKENS["eos"]]

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Eval", leave=False):
            images = images.to(device)
            captions = captions.to(device)

            features = model.encoder(images)
            pred_ids = model.decoder.sample(features, sos_id=sos_id, eos_id=eos_id, max_len=CFG.max_caption_len)

            # references: list of list of tokens
            for i in range(images.size(0)):
                ref = vocab.decode(captions[i].tolist()).split()
                hyp = vocab.decode(pred_ids[i].tolist()).split()
                references.append([ref])
                hypotheses.append(hyp)

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu4

def train():
    seed_everything(CFG.seed)
    device = CFG.device

    # Build / load vocab
    if os.path.exists(CFG.vocab_path):
        vocab = Vocabulary.load(CFG.vocab_path)
    else:
        captions = load_all_captions(CFG.train_ann)
        vocab = Vocabulary(min_freq=CFG.min_word_freq)
        vocab.build(captions)
        vocab.save(CFG.vocab_path)

    train_ds = CocoCaptionDataset(CFG.train_img_dir, CFG.train_ann, vocab, CFG.max_caption_len, is_train=True)
    val_ds = CocoCaptionDataset(CFG.val_img_dir, CFG.val_ann, vocab, CFG.max_caption_len, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_size=CFG.embed_size, freeze=CFG.freeze_cnn)
    decoder = DecoderRNN(vocab_size=len(vocab), embed_size=CFG.embed_size, hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, dropout=CFG.dropout)
    model = CaptioningModel(encoder, decoder).to(device)

    pad_id = vocab.stoi[SPECIAL_TOKENS["pad"]]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.lr)

    best_bleu4 = -1.0
    start_epoch = 0
    if os.path.exists(CFG.ckpt_path):
        ckpt = load_checkpoint(CFG.ckpt_path, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_bleu4 = ckpt.get("best_score", -1.0)

    for epoch in range(start_epoch, CFG.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{CFG.epochs}")

        total_loss = 0.0
        for images, captions in pbar:
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)  # (B,T,V)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        bleu1, bleu4 = compute_bleu(model, val_loader, vocab, device)

        print(f"[Epoch {epoch+1}] loss={avg_loss:.4f} BLEU-1={bleu1:.4f} BLEU-4={bleu4:.4f}")

        # Save best
        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            save_checkpoint(CFG.ckpt_path, model, optimizer, epoch, best_bleu4)
            print(f"✅ Saved best checkpoint: {CFG.ckpt_path} (BLEU-4={best_bleu4:.4f})")

    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        # quick eval
        vocab = Vocabulary.load(CFG.vocab_path)
        val_ds = CocoCaptionDataset(CFG.val_img_dir, CFG.val_ann, vocab, CFG.max_caption_len, is_train=False)
        val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)

        encoder = EncoderCNN(embed_size=CFG.embed_size, freeze=True)
        decoder = DecoderRNN(vocab_size=len(vocab), embed_size=CFG.embed_size, hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, dropout=CFG.dropout)
        model = CaptioningModel(encoder, decoder).to(CFG.device)

        load_checkpoint(CFG.ckpt_path, model)
        bleu1, bleu4 = compute_bleu(model, val_loader, vocab, CFG.device)
        print(f"BLEU-1={bleu1:.4f}, BLEU-4={bleu4:.4f}")

if __name__ == "__main__":
    main()
