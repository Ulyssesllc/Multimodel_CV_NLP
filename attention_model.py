from model import  ImageEncoder, TextEncoder
import torch   
import torch.nn as nn 

class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
    def forward(self, query, key, value, key_padding_mask = None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights
class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps = 1e-6):
        super(AddNorm, self).__init__() 
        self.embed_dim = embed_dim 
        self.norm = nn.LayerNorm(embed_dim, eps = eps)
    def forward(self, x, sublayer_output):
        return self.norm(x+sublayer_output)
class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_fc = nn.Linear(768,512)   # text_vector to size 512
        self.fc_layer = nn.Linear(512,512)
        self.final_fc = nn.Linear(512,10)
    ## Multihead Attention ##
        self.cross_attn = Cross_MHA(512,8)
        self.self_attn = Cross_MHA(512,8)
        self.add_norm1= AddNorm(512)
        self.add_norm2 = AddNorm(512)
        self.relu = nn.ReLU()
        self.sofmax = nn.Softmax(dim=-1)
        
    def forward(self,img, text):
        img = self.img_encoder(img)
        text = self.text_encoder(text)
        text = self.text_fc(text)
        
        text_cross, _ = self.cross_attn(query = img, key = text, value = text)
        img_cross, _ = self.cross_attn(query = text, key = img, value = img)
        combined = torch.cat((text_cross, img_cross), dim=1)  # [batch, 1024]
        combined = nn.Linear(1024, 512).to(combined.device)(combined)
        
        # Self Multi-Head Attention cho vector kết hợp
        self_attn_output, _ = self.self_attn(query=combined, key=combined, value=combined)
        
        # Add & Norm
        combined = self.add_norm1(combined, self_attn_output)
        
        # Feed Forward Layer
        ff_output = self.fc_layer(combined)
        ff_output = self.relu(ff_output)
        combined = self.add_norm2(combined, ff_output)
        
        # Lớp phân loại cuối cùng
        output = self.final_fc(combined)
        return output
        
        