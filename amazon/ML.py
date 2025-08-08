import os
from MLP import FusionModule
import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData
from torch.utils.data import random_split

from torch.utils.data import DataLoader
import copy

base_dir = os.path.dirname(__file__)
csv_file = os.path.join(base_dir, "training_data.csv")
# Attempt to find image directory under amazon/, else fallback to root folder
candidate = os.path.join(base_dir, "amazon_dataset")
if os.path.isdir(candidate):
    img_dir = candidate
else:
    img_dir = os.path.join(os.path.dirname(base_dir), "amazon_dataset")
dataset = MyData(csv_file, img_dir)
# determine number of classes dynamically
num_classes = len(dataset.label_map)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_data = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=8, shuffle=False)

# initialize model with correct output size
model = FusionModule(num_classes=num_classes)


def train(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        # training
        model.train()
        total_loss = 0
        for batch in train_loader:
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
        avg_train_loss = total_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                output = model(img, input_ids, attention_mask)
                loss = criterion(output, label)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # Run training workflow
    trained_model = train(model, train_data, test_data, epochs=50, lr=0.001, patience=5)
    # Save best model
    torch.save(trained_model.state_dict(), "fusion_model_best.pth")
