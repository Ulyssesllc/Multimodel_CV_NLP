import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import re
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
        # build label map dynamically from CSV label column
        # after filtering valid image files
        self.max_length = 512
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # Read CSV with latin1 encoding to handle special characters
        df = pd.read_csv(self.csv_file, sep=";", encoding="latin1")
        # Drop rows without img_path to avoid NaN values
        df = df.dropna(subset=["img_path"])
        # Extract filename from img_path using regex to handle both slashes
        df["img_filename"] = (
            df["img_path"].astype(str).apply(lambda p: re.split(r"[\\\\/]", p)[-1])
        )
        df = df[
            df["img_filename"].apply(
                lambda fn: os.path.isfile(os.path.join(self.img_file, fn))
            )
        ].reset_index(drop=True)
        # dynamic mapping: unique label texts to indices
        unique_labels = sorted(df["label"].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        # add numeric label index column
        df["label_index"] = df["label"].map(self.label_map)
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
        # numeric label index from dynamic map
        label = torch.tensor(int(item["label_index"]))
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": label,
            # include original description text for visualization
            "description": text,
        }
