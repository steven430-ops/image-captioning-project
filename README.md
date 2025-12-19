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

- **Training Strategy**
  - Teacher Forcing
  - Cross-Entropy Loss
  - Adam Optimizer

- **Evaluation**
  - BLEU-1, BLEU-4 scores
  - Qualitative comparison between generated captions and ground truth captions

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
⚙️ Installation


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

