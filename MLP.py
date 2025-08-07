import torch.nn as nn 
import torch
import torchvision.models as models 
from transformers import BertModel, BertTokenizer

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained = True)
        self.resnet.fc = nn.Identity()
    def forward(self,x):
        return self.resnet(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
    def forward(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)
        output = self.bert(input_ids = tokens["input_ids"], attention_mask = tokens["attention_mask"])
        return output.last_hidden_state[:, 0, :]
class MLP_fusion(nn.Module):
    def __init__(self):
        super(MLP_fusion, self).__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_fc = nn.Linear(768,512)   # text_vector to size 512
        self.fusion = nn.Linear(1024,512)
        self.fc_layer = nn.Linear(512,512)
        self.final_fc = nn.Linear(512,10)
        
    def forward(self, img, text):
        img = self.img_encoder(img)
        text = self.text_encoder(text)
        text = self.text_fc(text)
        combined = torch.cat((img, text), dim = 1)
        combined = self.fusion(combined)
        combined = self.fc_layer(combined)
        output = self.final_fc(combined)
        return output

        
