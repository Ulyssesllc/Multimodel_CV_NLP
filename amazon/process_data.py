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
        df = pd.read_csv(self.csv_file, sep=";", encoding="latin1")
        # Drop rows without img_path to avoid NaN values
        df = df.dropna(subset=["img_path"])
        # Extract filename from img_path and keep only valid image files
        df["img_filename"] = df["img_path"].apply(lambda p: os.path.basename(str(p)))
        df = df[
            df["img_filename"].apply(
                lambda fn: os.path.isfile(os.path.join(self.img_file, fn))
            )
        ].reset_index(drop=True)
        self.dataset = df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get row data
        item = self.dataset.iloc[idx]
        # Load image using img_path column (relative to img_file)
        # Use precomputed filename
        img_filename = item["img_filename"]
        img_path = os.path.join(self.img_file, img_filename)
        # Fallback to blank image if file missing
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # create a black image
            image = Image.new("RGB", (224, 224), (0, 0, 0))
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
        # Convert to 0-based class index
        label = torch.tensor(int(item["label_id"]) - 1)
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": label,
        }
