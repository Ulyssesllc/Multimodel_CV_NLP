import torch.nn as nn
import torch
import torchvision.models as models
from transformers import BertModel
from config import CONFIG


class ImageEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", freeze: bool = False):
        super(ImageEncoder, self).__init__()
        if backbone == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
        else:
            self.resnet = models.resnet18(pretrained=True)
        self.out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        if freeze:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


class TextEncoder(nn.Module):
    def __init__(self, freeze: bool = False):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.bert.device)
        attention_mask = attention_mask.to(self.bert.device)
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]


class FusionModule(nn.Module):
    def __init__(self, num_classes=10, dropout: float | None = None):
        super(FusionModule, self).__init__()
        self.img_encoder = ImageEncoder(
            backbone=CONFIG.backbone, freeze=CONFIG.freeze_backbone
        )
        self.text_encoder = TextEncoder(freeze=CONFIG.freeze_text)
        self.text_fc = nn.Linear(768, 512)
        fusion_in = self.img_encoder.out_dim + 512
        self.fusion = nn.Linear(fusion_in, 512)
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout if dropout is not None else CONFIG.dropout),
        )
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
