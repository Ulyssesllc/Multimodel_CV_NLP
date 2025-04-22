from MLP import FusionModule
import torch
import torch.optim as optim
import torch.nn as nn
from process_data import MyData 
from torch.utils.data import random_split
from torch.utils.data import DataLoader


def train(model, csv_file, img_dir, epochs):
    dataset = MyData(csv_file, img_dir) 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_data = DataLoader(train_dataset, batch_size= 8, shuffle= True)
    test_data = DataLoader(test_dataset, batch_size = 8, shuffle= False )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([
    {
        'params': model.text_encoder.bert.parameters(),
        'lr': 2e-5,
        'weight_decay': 0.01
    },
    {
        'params': model.q_mlp.parameters(),
        'lr': 1e-3,
        'weight_decay': 0.01
    },
    {
        'params': model.fusion.parameters(),
        'lr': 1e-3,
        'weight_decay': 0.01
    },
    {
        'params': model.cross_Q.parameters(),
        'lr': 1e-3,
        'weight_decay': 0.01
    }
])

    for epoch in range(epochs):
        model.train()
        total_loss = 0 
        correct = 0
        for batch in train_data:
            img = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            optimizer.zero_grad()
            output, logit_img, logit_text = model(img, input_ids, attention_mask)
            loss = criterion(output, label) + compute_itc_loss(logit_img, logit_text)
            
            prediction = output.argmax(dim = 1)
            correct+= (prediction == label).sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        result_test = test(model, test_data)
        torch.save(model.state_dict(), f'epoch_{epoch}.pth')
        # print(f"Epoch {epoch+1} / {epochs}, Loss:{total_loss/ len(dataloader)}, Accuracy: {correct/train_size}, Test_accuracy: {result_test}")
        log_message = f"Epoch {epoch+1} / {epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {correct / train_size:.4f}, Test_accuracy: {result_test:.4f}"
        print(log_message)

# Save to log file
        with open("training_log.txt", "a") as f:
            f.write(log_message + "\n")
def test(model, dataloader):
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
            output, logit_img, logit_text = model(img, input_ids, attention_mask)
            predictions = output.argmax(dim=1)  # Get predicted labels
            correct += (predictions == label).sum().item()
    return correct/test_size
            
    
