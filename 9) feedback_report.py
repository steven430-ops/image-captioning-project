import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import CFG
from dataset import CocoCaptionDataset
from model import EncoderCNN, DecoderRNN, CaptioningModel
from utils import Vocabulary, load_checkpoint, collate_fn, SPECIAL_TOKENS

def compute_bleu_and_samples(model, loader, vocab, device, num_samples=5):
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    references, hypotheses = [], []
    samples = []

    sos_id = vocab.stoi[SPECIAL_TOKENS["sos"]]
    eos_id = vocab.stoi[SPECIAL_TOKENS["eos"]]

    with torch.no_grad():
        for images, captions in tqdm(loader, desc="Reporting", leave=False):
            images = images.to(device)
            captions = captions.to(device)

            features = model.encoder(images)
            pred_ids = model.decoder.sample(features, sos_id=sos_id, eos_id=eos_id, max_len=CFG.max_caption_len)

            for i in range(images.size(0)):
                ref_txt = vocab.decode(captions[i].tolist())
                hyp_txt = vocab.decode(pred_ids[i].tolist())

                references.append([ref_txt.split()])
                hypotheses.append(hyp_txt.split())

                if len(samples) < num_samples:
                    samples.append((ref_txt, hyp_txt))

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu4, samples

def main():
    Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.load(CFG.vocab_path)
    val_ds = CocoCaptionDataset(CFG.val_img_dir, CFG.val_ann, vocab, CFG.max_caption_len, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_size=CFG.embed_size, freeze=True)
    decoder = DecoderRNN(vocab_size=len(vocab), embed_size=CFG.embed_size, hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, dropout=CFG.dropout)
    model = CaptioningModel(encoder, decoder).to(CFG.device)

    load_checkpoint(CFG.ckpt_path, model)

    bleu1, bleu4, samples = compute_bleu_and_samples(model, val_loader, vocab, CFG.device)

    report_path = os.path.join(CFG.out_dir, "feedback_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Feedback Report\n\n")
        f.write(f"- BLEU-1: **{bleu1:.4f}**\n")
        f.write(f"- BLEU-4: **{bleu4:.4f}**\n\n")
        f.write("## Sample Outputs\n\n")
        for idx, (ref, hyp) in enumerate(samples, 1):
            f.write(f"### Sample {idx}\n")
            f.write("**[Ground Truth]**\n\n")
            f.write(f"{ref}\n\n")
            f.write("**[Prediction]**\n\n")
            f.write(f"{hyp}\n\n")

    print(f"✅ Saved report: {report_path}")

if __name__ == "__main__":
    main()
