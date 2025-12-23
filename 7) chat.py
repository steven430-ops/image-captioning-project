import argparse
import torch
from PIL import Image
import torchvision.transforms as T

from config import CFG
from model import EncoderCNN, DecoderRNN, CaptioningModel
from utils import Vocabulary, load_checkpoint, SPECIAL_TOKENS

def load_model():
    vocab = Vocabulary.load(CFG.vocab_path)

    encoder = EncoderCNN(embed_size=CFG.embed_size, freeze=True)
    decoder = DecoderRNN(vocab_size=len(vocab), embed_size=CFG.embed_size, hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, dropout=CFG.dropout)
    model = CaptioningModel(encoder, decoder).to(CFG.device)

    load_checkpoint(CFG.ckpt_path, model)
    model.eval()
    return model, vocab

def preprocess_image(img_path: str):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

@torch.no_grad()
def predict(model, vocab, img_path: str):
    image = preprocess_image(img_path).to(CFG.device)
    features = model.encoder(image)

    sos_id = vocab.stoi[SPECIAL_TOKENS["sos"]]
    eos_id = vocab.stoi[SPECIAL_TOKENS["eos"]]
    pred_ids = model.decoder.sample(features, sos_id=sos_id, eos_id=eos_id, max_len=CFG.max_caption_len)[0].tolist()
    return vocab.decode(pred_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="path to an image file")
    args = parser.parse_args()

    model, vocab = load_model()
    caption = predict(model, vocab, args.image)
    print("Prediction:", caption)

if __name__ == "__main__":
    main()
