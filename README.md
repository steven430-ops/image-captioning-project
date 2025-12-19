
# Image Captioning Project

CNN-RNN based Image Captioning

---

## 🎯 Objective

The goal of this project is to build an **Image Captioning model** that automatically generates a natural language description for a given image.  
The model extracts visual features using a **Convolutional Neural Network (CNN)** and generates captions using a **Recurrent Neural Network (RNN, LSTM)**.

---

## 🧠 Approach

- **Dataset**
  - MS COCO Captions Dataset

- **Model Architecture**
  - CNN Encoder: ResNet50 (pretrained)
  - RNN Decoder: LSTM
  - (Optional) Attention Mechanism

### 🔧 Core Model Architecture (Encoder–Decoder)

```python
class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
The model follows an encoder–decoder architecture, where a CNN extracts visual
features from images and an LSTM generates captions sequentially based on those features.


- **Training Strategy**
  - Teacher Forcing
  - Cross-Entropy Loss
  - Adam Optimizer

### 🏋️ Training with Teacher Forcing

```python
captions_input = captions[:, :-1]
captions_target = captions[:, 1:]

outputs = model(images, captions_input)
loss = criterion(
    outputs.reshape(-1, outputs.size(-1)),
    captions_target.reshape(-1)
)
During training, teacher forcing is applied by shifting the input and target
caption sequences, enabling stable and efficient learning.

- **Evaluation**
  - BLEU-1, BLEU-4 scores
  - Qualitative comparison between generated captions and ground truth captions

### 📐 BLEU Score Evaluation

```python
from nltk.translate.bleu_score import corpus_bleu

bleu4 = corpus_bleu(
    references,
    hypotheses,
    weights=(0.25, 0.25, 0.25, 0.25)
)
BLEU-4 measures n-gram overlap between generated captions and ground truth captions,
providing a quantitative evaluation of caption quality.

---

## 📁 Project Structure

```text
image-captioning-project/
├── README.md
├── config.py
├── utils.py
├── dataset.py
├── model.py
├── main.py
└── results/

---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio
pip install pillow tqdm numpy nltk pycocotools

🧩 File Descriptions
1. config.py

Manages dataset paths and hyperparameters

Centralized configuration for training and model settings

2. utils.py

Caption preprocessing and tokenization

Vocabulary construction and padding

3. dataset.py

Loads image-caption pairs from MS COCO

Implements PyTorch Dataset class

4. model.py

CNN Encoder (ResNet50)

LSTM Decoder for caption generation

5. main.py

Training loop

BLEU score evaluation

Sample caption generation

📊 Results
Metric	Score
BLEU-1	0.62
BLEU-4	0.29

BLEU scores are reported on the validation set using greedy decoding.

🖼️ Sample Output
[Ground Truth]
A man riding a skateboard on a city street.

[Prediction]
A person riding a skateboard down the street.

🔍 Future Work

Integrate attention mechanism

Apply beam search decoding

Evaluate using CIDEr and METEOR metrics

📝 Conclusion

This project demonstrates the effectiveness of combining CNN-based visual feature extraction with RNN-based sequence generation for image captioning tasks.
Through this work, we gained practical experience in multimodal deep learning and sequence-to-sequence modeling.
