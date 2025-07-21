import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData 
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from Contrastive import Q_cons_fusion, compute_itc_loss
from torch.optim import AdamW, Adam
import os
from MLP import MLP_fusion
from torch.cuda.amp import autocast, GradScaler
from attention_model import AttentionModule
from tqdm import tqdm
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import os
import multiprocessing
import argparse
import pandas as pd
os.makedirs("checkpoints", exist_ok=True)
df = pd.read_csv('GLAMI-1M-train.csv')
label_map = {label: idx for idx, label in enumerate(df['category_name'].unique())}
# print(label_map)
train_dataset = MyData('GLAMI-1M-train.csv', 'images', label_map)
test_dataset = MyData('GLAMI-1M-test.csv', 'images', label_map)

def args():
    parser = argparse.ArgumentParser(description="Training script for multimodal model")
    parser.add_argument('--model', type=str, default='Q_cons_fusion', help='Model to use: Q_cons_fusion or MLP_fusion')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    return parser.parse_args()

# train_size = int(0.8 * len(df))

# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])
print(multiprocessing.cpu_count())
train_data = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
test_data = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

print('hi')

def train(model, dataloader, epochs = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
#     optimizer = AdamW([
#     {
#         'params': model.module.text_encoder.bert.parameters(),  # thêm .module
#         'lr': 2e-5,
#         'weight_decay': 0.01
#     },
#     {
#         'params': model.module.q_mlp.parameters(),
#         'lr': 5e-4,
#         'weight_decay': 0.01
#     },
#     {
#         'params': model.module.fusion.parameters(),
#         'lr': 5e-4,
#         'weight_decay': 0.01
#     },
#     {
#         'params': model.module.cross_Q.parameters(),
#         'lr': 1e-3,
#         'weight_decay': 0.01
#     }
    
# ])
    if hasattr(model, 'module'):
        params = model.module.parameters()
    else:
        params = model.parameters()
    optimizer = AdamW(params, lr=1e-4, weight_decay=0.01)
        # scaler = GradScaler()

    best_acc = 0 
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                output, logit_img, logit_text = model(img, input_ids, attention_mask)
                loss = criterion(output, label) + compute_itc_loss(logit_img, logit_text)
            
            prediction = output.argmax(dim = 1)
            correct+= (prediction == label).sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)  # cần thiết để clip_grad_norm đúng
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # scaler.step(optimizer)
            # scaler.update()
            total_loss += loss.item()
        result_test = test(model, test_data)
        
        log_message = f"Epoch {epoch+1} / {epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / len(train_dataset):.4f}, Test_accuracy: {result_test:.4f}"
        print(log_message)
        if (result_test>best_acc):
            best_acc = result_test
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"Saved best model at epoch {epoch+1} with test accuracy {best_acc:.4f}")
        with open("-constrastive.txt", "a") as f:
            f.write(log_message + "\n")
def test(model, dataloader, check = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            if check == True:
                output, _, _ = model(img, input_ids, attention_mask)
            else:
                output = model(img, input_ids, attention_mask)
            predictions = output.argmax(dim=1)  # Get predicted labels
            correct += (predictions == label).sum().item()
    return correct/len(test_dataset)

def train1(model, dataloader, epochs = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(img, input_ids, attention_mask)
                loss = criterion(output, label) 
            
            prediction = output.argmax(dim = 1)
            correct+= (prediction == label).sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        result_test = test(model, test_data, check = False)

        log_message = f"Epoch {epoch+1} / {epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / len(train_dataset):.4f}, Test_accuracy: {result_test:.4f}, Test_size: {len(test_dataset)}"
        print(log_message)

# Save to log file
        with open("attention.txt", "a") as f:
            f.write(log_message + "\n")

if __name__ == "__main__":
    args = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == 'Q_cons_fusion':
        model = Q_cons_fusion()
        model = torch.nn.DataParallel(model)
        model.to(device)
        # print(model)
        train(model, train_data)
        
    else:
        model = MLP_fusion()
        model = torch.nn.DataParallel(model)
        model.to(device)
        # print(model)
        train1(model, train_data)



    
