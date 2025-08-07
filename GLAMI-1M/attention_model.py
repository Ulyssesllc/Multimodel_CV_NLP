(
    """""
1) Lớp Cross_MHA

Bọc trên nn.MultiheadAttention của PyTorch để thực hiện attention giữa hai luồng (ví dụ: văn bản → hình ảnh hoặc ngược lại).
Tham số chính là embed_dim (kích thước embedding) và num_heads (số đầu attention).

2) Lớp AddNorm

Thực hiện phép cộng skip-connection (x + sublayer_output) rồi chuẩn hóa (LayerNorm).
Giúp ổn định quá trình huấn luyện sau mỗi sub-layer (attention hoặc feed-forward).

3) Lớp AttentionModule

Khởi tạo hai encoder:
• ImageEncoder (MLP) để trích xuất feature từ ảnh đầu vào;
• TextEncoder (MLP) để trích xuất feature từ chuỗi token.
Dùng các layer Linear để đưa feature ảnh (2048-dim) và văn bản (768-dim) về cùng không gian 512-dim, rồi gộp lại thành 1024-dim.
Thực hiện:
a) Cross-attention:
Text attends vào image và image attends vào text (qua hai instance cross_attn).
b) Mix & Dropout: kết hợp hai luồng, rồi nối thành vector chung 512-dim.
c) Self-attention: attention nội bộ trên vector kết hợp (qua self_attn).
d) Add & Norm: skip-connection và LayerNorm sau self-attention.
e) Feed-forward: một lớp Linear + ReLU + dropout, rồi Add & Norm tiếp.
f) Classification: hai lớp Linear cuối để giảm xuống 4 chiều (giả sử 4 lớp phân loại).

4) Luồng dữ liệu (forward)

Đầu vào: ảnh (img) và token văn bản (input_ids, attention_mask).
Encode –> biến đổi về embedding 512 –> thêm chiều thứ hai (batch, 1, 512) –> cross-attention hai chiều –> concat và mix –> self-attention –> feed-forward –> dự đoán đầu ra (logits 4 chiều).
Nguyên lý hoạt động chính dựa trên Transformer-style multi-head attention kết hợp thông tin đa modal (ảnh + văn bản), sử dụng cơ chế skip-connection+LayerNorm để giữ ổn định và cuối cùng là một mạng fully-connected cho phân lớp.
"""
    ""
)

from MLP import ImageEncoder, TextEncoder
import torch
import torch.nn as nn


class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Cross_MHA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask)
        return attn_output, attn_weights


class AddNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(AddNorm, self).__init__()
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.img_fc = nn.Linear(2048, 512)
        self.text_fc = nn.Linear(768, 512)  # text_vector to size 512
        self.mix = nn.Linear(1024, 512)
        self.fc_layer = nn.Linear(512, 512)
        self.final_g = nn.Linear(512, 128)
        self.final_fc = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(512)

        #    self.dropout = nn.Dropout(p=0.3)
        # self.norm1 = nn.BatchNorm1d(512)
        # self.norm2 = nn.BatchNorm1d(256)
        ## Multihead Attention ##
        self.cross_attn = Cross_MHA(512, 8)
        self.self_attn = Cross_MHA(512, 8)
        self.add_norm1 = AddNorm(512)
        self.add_norm2 = AddNorm(512)
        self.relu = nn.ReLU()
        self.sofmax = nn.Softmax(dim=-1)

    def forward(self, img, input_ids, attention_mask):
        img = self.img_encoder(img)
        img = self.relu(self.norm1(self.img_fc(img)))
        text = self.text_encoder(input_ids, attention_mask)
        text = self.relu(self.norm1(self.text_fc(text)))
        text = text.unsqueeze(1)  # (batch, 1, 512)
        img = img.unsqueeze(1)  # (batch, 1, 512)

        text_cross, _ = self.cross_attn(query=img, key=text, value=text)
        img_cross, _ = self.cross_attn(query=text, key=img, value=img)
        combined = torch.cat(
            (text_cross.squeeze(1), img_cross.squeeze(1)), dim=1
        )  # [batch, 1024]
        combined = self.relu(self.dropout(self.mix(combined)))
        combined = combined.unsqueeze(1)  # (batch, 1, 512)

        # Self Multi-Head Attention cho vector kết hợp
        self_attn_output, _ = self.self_attn(
            query=combined, key=combined, value=combined
        )

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
