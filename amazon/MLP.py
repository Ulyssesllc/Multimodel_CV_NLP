import torch.nn as nn
import torch
import torchvision.models as models
from transformers import BertModel


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]


class FusionModule(nn.Module):
    def __init__(self, num_classes=10):
        super(FusionModule, self).__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_fc = nn.Linear(768, 512)  # text_vector to size 512
        self.fusion = nn.Linear(1024, 512)
        self.fc_layer = nn.Linear(512, 512)
        # final classification layer with dynamic output size
        self.final_fc = nn.Linear(512, num_classes)

    def forward(self, img, input_ids, attention_mask):
        img_feat = self.img_encoder(img)
        text_feat = self.text_encoder(input_ids, attention_mask)
        text_feat = self.text_fc(text_feat)
        combined = torch.cat((img_feat, text_feat), dim=1)
        combined = self.fusion(combined)
        combined = self.fc_layer(combined)
        output = self.final_fc(combined)
        return output
