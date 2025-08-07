import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import BertTokenizer


class MyData(Dataset):
    def __init__(self, csv_file, img_file):
        self.csv_file = csv_file
        self.img_file = img_file
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.label_map = {
            "Action": 0,
            "Comedy": 1,
            "Horror": 2,
            "Romance": 3,
        }
        self.max_length = 512
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Read CSV with latin1 encoding to handle special characters
        self.dataset = pd.read_csv(self.csv_file, sep=";", encoding="latin1")
        # self.max_length
        # self.dataset = pd.read_csv(self.csv_file, encoding="latin1")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get row data
        item = self.dataset.iloc[idx]
        # Load image using img_path column (relative to img_file)
        # Extract filename to match files in img_file directory
        img_filename = os.path.basename(item["img_path"])
        img_path = os.path.join(self.img_file, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Tokenize text description
        text = str(item["description"])
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Use numeric label_id column
        label = torch.tensor(int(item["label_id"]))
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": label,
        }
