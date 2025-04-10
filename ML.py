from MLP import FusionModule
import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData 
from torch.utils.data import random_split

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# def __init__(self, csv_file, img_file):
csv_file = "training_data.csv"
img_dir = "amazon_dataset"
dataset = MyData(csv_file, img_dir)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_data = DataLoader(train_dataset, batch_size= 8, shuffle= True)
test_data = DataLoader(test_dataset, batch_size = 8, shuffle= False )

feature_extraction = FusionModule()

def train(feature_extraction, dataloader, epochs = 15, lr = 0.001, model = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extraction.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        feature_extraction.train()
        total_loss = 0 
        for batch in dataloader:
            img = batch['image'].to(device)
            text = batch['text'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            output = feature_extraction(img, text)   # output là feature vector có số chiều là 512
            output = model(output)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} / {epochs}, Loss:{total_loss/ len(dataloader)}")
    
