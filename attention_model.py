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
        self.img_fc = nn.Linear(2048,512)
        self.text_fc = nn.Linear(768,512)   # text_vector to size 512
        self.mix = nn.Linear(1024,512)
        self.fc_layer = nn.Linear(512,512)
        self.final_g = nn.Linear(512,128)
        self.final_fc = nn.Linear(128,4)
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(512)
        
        #    self.dropout = nn.Dropout(p=0.3)
        # self.norm1 = nn.BatchNorm1d(512)
        # self.norm2 = nn.BatchNorm1d(256) 
    ## Multihead Attention ##
        self.cross_attn = Cross_MHA(512,8)
        self.self_attn = Cross_MHA(512,8)
        self.add_norm1= AddNorm(512)
        self.add_norm2 = AddNorm(512)
        self.relu = nn.ReLU()
        self.sofmax = nn.Softmax(dim=-1)
        
    def forward(self,img, input_ids, attention_mask):
        img = self.img_encoder(img)
        img = self.relu(self.norm1(self.img_fc(img)))
        text = self.text_encoder(input_ids, attention_mask)
        text = self.relu(self.norm1(self.text_fc(text)))
        text = text.unsqueeze(1)  # (batch, 1, 512)
        img = img.unsqueeze(1)    # (batch, 1, 512)

        text_cross, _ = self.cross_attn(query = img, key = text, value = text)
        img_cross, _ = self.cross_attn(query = text, key = img, value = img)
        combined = torch.cat((text_cross.squeeze(1), img_cross.squeeze(1)), dim=1)  # [batch, 1024]
        combined = self.relu(self.dropout(self.mix(combined)))
        combined = combined.unsqueeze(1)  # (batch, 1, 512)

        # Self Multi-Head Attention cho vector kết hợp
        self_attn_output, _ = self.self_attn(query=combined, key=combined, value=combined)
        
        # Add & Norm
        combined = self.add_norm1(combined.squeeze(1), self_attn_output.squeeze(1))
        
        # Feed Forward Layer
        ff_output = self.dropout(self.fc_layer(combined))
        ff_output = self.relu(ff_output)
        combined = self.add_norm2(combined, ff_output)
        
        # Lớp phân loại cuối cùng
        output = self.relu(self.dropout(self.final_g(combined)))
        output = self.final_fc(output)
        return output
        
        