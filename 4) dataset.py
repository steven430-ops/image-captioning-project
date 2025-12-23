# 4) dataset.py
from typing import List, Tuple
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as T

from utils import Vocabulary

class CocoCaptionDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, vocab: Vocabulary, max_len: int, is_train: bool = True):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.vocab = vocab
        self.max_len = max_len

        # annotation ids
        self.ann_ids = list(self.coco.anns.keys())

        # transforms
        if is_train:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.ann_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann_id = self.ann_ids[idx]
        ann = self.coco.anns[ann_id]
        caption = ann["caption"]

        img_info = self.coco.loadImgs(ann["image_id"])[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        cap_ids = self.vocab.encode(caption, self.max_len)
        cap_ids = torch.tensor(cap_ids, dtype=torch.long)

        return image, cap_ids

def load_all_captions(ann_file: str) -> List[str]:
    coco = COCO(ann_file)
    caps = [a["caption"] for a in coco.anns.values()]
    return caps
