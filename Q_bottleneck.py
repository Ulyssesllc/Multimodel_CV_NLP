import os
import sys
import numpy as np 
from transformers import BertModel
import torch.nn as nn
import timm
import torch.nn.functional as F
import torch
expert_usage = None
def init_expert(num_experts):
    global expert_usage
    expert_usage = torch.zeros(num_experts, dtype=torch.float32)
def update_expert_usage(mask):
    global expert_usage
    if expert_usage is None:
        return
    expert_usage += (mask>0).to(torch.long).sum(dim=0).cpu()
def get_expert_usage():
    global expert_usage
    if expert_usage is None:
        raise ValueError("Expert usage not initialized. Call init_expert first.")
    return expert_usage

class Text_encoder(nn.Module):
    def __init__(self):
        super(Text_encoder, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
    def forward(self, input_ids, attention_mask):
        output = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask)
        return output.last_hidden_state

class Image_encoder(nn.Module):
    def __init__(self):
        super(Image_encoder, self).__init__()
        self.image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.image_encoder.head = nn.Identity()
    def forward(self, img):
        output = self.image_encoder.forward_features(img)
        return output     
class MHA(nn.Module):
    def __init__(self, nums_head, d_embed):
        super(MHA, self).__init__()
        self.nums_head = nums_head
        self.d_embed = d_embed
        self.mha = nn.MultiheadAttention(d_embed, nums_head, batch_first = True)
    def forward(self, query, key, value, key_padding_mask = None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights

# class 

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 1. Cross-Attention
        attn_output, _ = self.attn(query, key, value)
        
        # 2. Add & Norm (Residual Connection 1)
        query = self.norm1(query + self.dropout(attn_output))
        
        # 3. Feed-Forward
        ffn_output = self.ffn(query)
        
        # 4. Add & Norm (Residual Connection 2)
        query = self.norm2(query + self.dropout(ffn_output))
        
        return query

class Gate_way(nn.Module):
    def __init__(self,embed_dim = 768):
        super(Gate_way, self).__init__()
        self.embed_dim = embed_dim
        self.gate = nn.Sequential(
            nn.Linear(2* embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, img, txt):
        assert img.shape == txt.shape
        new = torch.cat([img, txt], dim = -1)
        g = self.gate(new)
        fuse = img * g + (1-g) * txt
        return fuse

class MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 8,
        expert_capacity: int = 1,               # currently unused → remove or implement
        top_k: int = 2,                         # renamed from num_experts_per_token
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
        self.router =nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size,num_experts)
            )
          

        # Expert sub‑networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 4*hidden_size),
                nn.LayerNorm(4*hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(4*hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes)
            )
            for _ in range(num_experts)
        ])
        # get_expert_usage(self.num_experts)
    def forward(self, trunk_out1):
        # shape: [B, D]
        # print(trunk_out1.shape)
        B, num_input, len_embed = trunk_out1.size()
        trunk_out = trunk_out1.view(-1, len_embed)
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
        # if self.check_quality:
        update_expert_usage(mask)
        #    final_logits: [B, num_classes]
        final_logits = torch.zeros([trunk_out.size(0), self.num_classes], device=trunk_out.device)
        for expert_id in range(self.num_experts):
    # 2. Lấy index của samples route vào expert này
            mask_e = mask[:, expert_id]        # shape [B]
            selected_idx = (mask_e > 0).nonzero(as_tuple=False).squeeze()

            if selected_idx.numel() == 0:
                continue
            if selected_idx.ndim == 0:  
                selected_idx = selected_idx.unsqueeze(0)   # biến scalar -> [1]

            x_selected = trunk_out[selected_idx]                       # [S, D]
            weight_selected = mask_e[selected_idx].unsqueeze(1)        # [S, 1]
            output_selected = self.experts[expert_id](x_selected)      # [S, C]
            output_selected = output_selected * weight_selected
            final_logits[selected_idx] += output_selected
            
        final_logits = final_logits.view(B, num_input, self.num_classes)  # [B, num_input, C]
        final_logits = final_logits.mean(dim=1)
        return final_logits, aux_loss



#### Phase 2 ------  Q-transform    

class Q_bottleneck(nn.Module):
    def __init__(self):
        super(Q_bottleneck, self).__init__()
        self.text_encoder = Text_encoder()
        self.moe = MoE()
        self.image_encoder = Image_encoder()
        self.Q1 = nn.Parameter(torch.randn(32,768))
        self.Q2 = nn.Parameter(torch.randn(32,768))
        self.Q3 = nn.Parameter(torch.randn(2,768))
        self.num_layer = 6
        self.cross_attention_layers1 = nn.ModuleList(
            [CrossAttentionBlock(embed_dim= 768 , num_heads = 8, ff_dim=768*4) for _ in range(self.num_layer)]
        )
        self.cross_attention_layers2 = nn.ModuleList(
            [CrossAttentionBlock(embed_dim= 768 , num_heads = 8, ff_dim=768*4) for _ in range(self.num_layer)]
        )
        self.cross_attention_layers3 = nn.ModuleList(
            [CrossAttentionBlock(embed_dim= 768 , num_heads = 8, ff_dim=768*4) for _ in range(self.num_layer)]
        )
        self.gate_way = Gate_way()
        self.fusion = nn.Sequential(
            # nn.LayerNorm(hidden_dim * 2, eps =1e-6),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout to prevent early overfitting
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 191)
        )
    def forward(self, input_ids, attention_mask, image):
        B = image.size(0)
        Q1_tokens = self.Q1.unsqueeze(0).repeat(B, 1, 1)             # B x 32 x 768
        Q2_tokens = self.Q2.unsqueeze(0).repeat(B, 1, 1)
        final_fuse = self.Q3.unsqueeze(0).repeat(B,1,1)
        txt = self.text_encoder(input_ids, attention_mask)   # B x 512 x 768
        img = self.image_encoder(image)        # Bx197x768
        for block in self.cross_attention_layers1:
            Q1_tokens = block(query=Q1_tokens, key= img, value=img)  # B x 32 x 768
        for block in self.cross_attention_layers2:
            Q2_tokens = block(query=Q2_tokens, key = txt, value =txt)
        fuse = self.gate_way(Q1_tokens, Q2_tokens)
        for block in self.cross_attention_layers3:
            final_fuse = block(query = final_fuse, key = fuse, value = fuse)  # B x 2 x 768
        # final_fuse = final_fuse.mean(dim=1)
        # final = self.fusion(final_fuse)
        logit_final, aux_loss = self.moe(final_fuse)
        return logit_final, aux_loss,  Q1_tokens.mean(dim=1), Q2_tokens.mean(dim=1)
    
def compute_itc_loss(img_feat, text_feat, temperature=0.07):
    """
    img_feat: Tensor [B, D]
    text_feat: Tensor [B, D]
    """
    # Normalize embeddings
    img_feat = F.normalize(img_feat, dim=-1, eps=1e-6)
    text_feat = F.normalize(text_feat, dim=-1, eps=1e-6)

    # Cosine similarity: [B, B]
    logits_per_image = img_feat @ text_feat.T
    logits_per_text = text_feat @ img_feat.T

    # Apply temperature
    logits_per_image /= temperature
    logits_per_text /= temperature

    # Ground-truth: diagonal (same index)
    targets = torch.arange(img_feat.size(0)).to(img_feat.device)

    # Cross entropy
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text, targets)
    
    # Final ITC loss
    loss = (loss_i2t + loss_t2i) / 2
    return loss
        
        
            
          
########   Phase 3:          


           