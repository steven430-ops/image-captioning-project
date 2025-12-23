import gradio as gr
import torch
from PIL import Image
import torchvision.transforms as T

from config import CFG
from model import EncoderCNN, DecoderRNN, CaptioningModel
from utils import Vocabulary, load_checkpoint, SPECIAL_TOKENS

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model():
    vocab = Vocabulary.load(CFG.vocab_path)

    encoder = EncoderCNN(embed_size=CFG.embed_size, freeze=True)
    decoder = DecoderRNN(vocab_size=len(vocab), embed_size=CFG.embed_size, hidden_size=CFG.hidden_size, num_layers=CFG.num_layers, dropout=CFG.dropout)
    model = CaptioningModel(encoder, decoder).to(CFG.device)

    load_checkpoint(CFG.ckpt_path, model)
    model.eval()
    return model, vocab

MODEL, VOCAB = load_model()

@torch.no_grad()
def caption_image(img: Image.Image):
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(CFG.device)
    features = MODEL.encoder(x)

    sos_id = VOCAB.stoi[SPECIAL_TOKENS["sos"]]
    eos_id = VOCAB.stoi[SPECIAL_TOKENS["eos"]]
    pred_ids = MODEL.decoder.sample(features, sos_id=sos_id, eos_id=eos_id, max_len=CFG.max_caption_len)[0].tolist()
    return VOCAB.decode(pred_ids)

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Caption"),
    title="Image Captioning Project (CNN + LSTM)",
    description="Upload an image and generate a caption using a CNN-RNN model."
)

if __name__ == "__main__":
    demo.launch()
