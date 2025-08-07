import os
from MLP import FusionModule
import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData
from torch.utils.data import random_split

from torch.utils.data import DataLoader

base_dir = os.path.dirname(__file__)
csv_file = os.path.join(base_dir, "training_data.csv")
# Attempt to find image directory under amazon/, else fallback to root folder
candidate = os.path.join(base_dir, "amazon_dataset")
if os.path.isdir(candidate):
    img_dir = candidate
else:
    img_dir = os.path.join(os.path.dirname(base_dir), "amazon_dataset")
dataset = MyData(csv_file, img_dir)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_data = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = FusionModule()


def train(model, dataloader, epochs=15, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            output = model(img, input_ids, attention_mask)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} / {epochs}, Loss:{total_loss / len(dataloader)}")
    return model


if __name__ == "__main__":
    # Run training workflow
    trained_model = train(model, train_data)
    # Optionally, save the trained model
    torch.save(trained_model.state_dict(), "fusion_model.pth")
