# File that contains the dataset and the collate_fn to use for the dataloader

import os
import pandas as pd
from torch.utils.data import Dataset, Subset
import ast
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")

class FakeNewsDataset(Dataset):
    def __init__(self, tsv_file, image_dir):
        self.data = pd.read_csv(tsv_file, sep='\t').drop_duplicates(keep="first", ignore_index=True)
        self.img_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        id = row["ID"]
        type = row["Type"]
        text = row["Text"]
        label = row["Label"]

        img_paths = ast.literal_eval(row["Media"])
        images = []
        for img_path in img_paths:
            total_path = os.path.join(self.img_dir, img_path)
            if not os.path.exists(total_path):
                continue
            image = pil_loader(os.path.join(self.img_dir, img_path))
            images.append(image)
        
        return {"id": id, "type": type, "text": text, "label": label, "images": images}
        

def collate_fn(batch):
    ids = []
    types = []
    texts = []
    labels = []
    images = []
    nums_images = []

    for sample in batch:
        images_list = sample["images"]
        num_images = len(images_list)
        nums_images.append(num_images)
        ids.append(sample["id"])
        types.append(sample["type"])
        texts.append(sample["text"])
        labels.append(sample["label"])
        for img in images_list:
            images.append(img)
   
    
    return {"id": ids, "type": types, "text": texts, "label": labels, "nums_images": nums_images, "images": images}
