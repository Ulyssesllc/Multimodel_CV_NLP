import torch
import torch.nn as nn
import timm
from transformers import BertModel
import torch.nn.functional as F

class Image_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.img_encoder.head = nn.Identity()  # Remove the classification head
    def forward(self,img):
        return self.img_encoder(img)
class Text_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]
class MHA(nn.Module):
    def __init__(self, num_heads, d_embed):
        super().__init__()
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.mha = nn.MultiheadAttention(d_embed, num_heads, batch_first=True)
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        return attn_output, attn_weights
class Cross_IT(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = Text_Encoder()
        self.image_encoder = Image_Encoder()
        self.mha_txt = MHA(num_heads = 8, d_embed = 768)
        self.mha_img = MHA(num_heads = 8, d_embed = 768)
    # Forward pass for cross-modal attention
    def forward(self, input_ids, attention_mask, img):
        text_output = self.text_encoder(input_ids, attention_mask)
        img_output = self.image_encoder(img)
        # Cross Attention
        text_update, _ = self.mha_txt(text_output, img_output, img_output)
        img_update, _ = self.mha_img(img_output, text_output, text_output)
        return torch.cat([text_update.squeeze(1), img_update.squeeze(1)], dim=1)
    
    
    
class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 64,
        expert_capacity: int = 1,               # currently unused → remove or implement
        top_k: int = 8,                         # renamed from num_experts_per_token
        jitter_noise: float = 0.01,             # unused → remove or add noise
        z_loss_coef: float = 0.1,
        load_balance_coef: float = 0.1,
        num_classes: int = 191
    ):
        super().__init__()
        # Store config
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.z_loss_coef = z_loss_coef
        self.load_balance_coef = load_balance_coef
        self.num_classes = num_classes

        # Routing head
        self.router = nn.Linear(2*hidden_size, num_experts)

        # Expert sub‑networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, num_classes)
            )
            for _ in range(num_experts)
        ])

        self.cross_it = Cross_IT()
    def forward(self, input_ids, attention_mask, image):
        # 1) Cross‑modal trunk
        trunk_out = self.cross_it(input_ids, attention_mask, image)  
        # shape: [B, D]

        # 2) Routing probabilities
        logits = self.router(trunk_out)           # [B, num_experts]
        probs = F.softmax(logits, dim=-1)         # [B, num_experts]

        # 3) Top‑k sparse mask
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # [B, top_k]
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        mask = torch.zeros_like(probs)
        mask.scatter_(1, topk_idx, topk_vals)    # [B, num_experts]

        # 4) Z‑loss (confidence regularizer)
        z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))
        z_loss = self.z_loss_coef * z_loss

        # 5) Load‑balancing loss
        load = mask.mean(dim=0)                  # [num_experts]
        ideal = 1.0 / self.num_experts
        lb_loss = ((load - ideal).pow(2)).sum()
        lb_loss = self.load_balance_coef * lb_loss

        # 6) Combine losses
        aux_loss = z_loss + lb_loss

        #    final_logits: [B, num_classes]
        final_logits = torch.zeros([trunk_out.size(0), self.num_classes], device=trunk_out.device)
        for i in range(self.num_experts):
            expert = self.experts[i]
            # Apply expert only to the top-k masked tokens
            # expert_outs: [B, num_classes]
            expert_output = expert(trunk_out)
            expert_mask = mask[:, i].unsqueeze(1)  # [B, 1]
            final_logits+= expert_output * expert_mask
            
        # expert_outs = torch.stack([expert(trunk_out) for expert in self.experts], dim=1)
        # final_logits = (expert_outs * mask.unsqueeze(-1)).sum(dim=1)

        return final_logits, aux_loss
