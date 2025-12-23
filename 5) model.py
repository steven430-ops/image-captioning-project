import torch
import torch.nn as nn
import torchvision.models as models

from utils import SPECIAL_TOKENS

class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, freeze: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone[0].weight.requires_grad):
            features = self.backbone(images)  # (B, 2048, 1, 1)
        features = features.flatten(1)        # (B, 2048)
        features = self.bn(self.fc(features)) # (B, embed_size)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        # captions: (B, T)
        embeddings = self.embed(captions)  # (B, T, E)
        # prepend image feature as first token input
        features = features.unsqueeze(1)   # (B, 1, E)
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)  # (B, T, E)
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)  # (B, T, V)
        return outputs

    @torch.no_grad()
    def sample(self, features: torch.Tensor, sos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
        # Greedy decoding
        B = features.size(0)
        inputs = features.unsqueeze(1)  # (B,1,E)
        states = None

        sampled_ids = []
        # first token is SOS
        prev = torch.full((B, 1), sos_id, dtype=torch.long, device=features.device)
        prev_emb = self.embed(prev)

        # first step: feed feature as "image token", then SOS token
        # We do two-step: image feature + sos embed by concatenation
        # simpler: run LSTM on [feature], then step using sos repeatedly
        h, states = self.lstm(inputs, states)
        out = self.fc(h.squeeze(1))
        # now loop tokens
        token = torch.argmax(out, dim=1)  # (B,)
        sampled_ids.append(token)

        for _ in range(max_len - 1):
            emb = self.embed(token.unsqueeze(1))
            h, states = self.lstm(emb, states)
            out = self.fc(h.squeeze(1))
            token = torch.argmax(out, dim=1)
            sampled_ids.append(token)

        sampled_ids = torch.stack(sampled_ids, dim=1)  # (B, max_len)
        return sampled_ids

class CaptioningModel(nn.Module):
    def __init__(self, encoder: EncoderCNN, decoder: DecoderRNN):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
