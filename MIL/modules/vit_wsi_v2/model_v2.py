
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn import functional as F, init
from torch import nn
import torch


class MHAPoolCell(nn.Module):
    def __init__(self, input_dim, num_heads=8, device='cpu', cls=False, num_classes=-1):
        super(MHAPoolCell, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        #self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        #self.attention2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        self.attention = CustomMultiHeadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1)
        self.attention2 = CustomMultiHeadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1)
        #self.pooling = nn.Parameter(torch.randn(1, 1, input_dim))
        
        
        #for name, param in self.attention.named_parameters():
        #    print(name, param.shape)

        #for name, param in self.attention2.named_parameters():
        #    print(name, param.shape)

        #assert 1 == 0

        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

        self.norm2 = nn.LayerNorm(input_dim)

        self.att_pool = MultiHeadAttentionPool(input_dim, 8)

        self.device = device
        self.cls = cls
        if cls:
            #assert 1 == 0
            assert num_classes > 0
            #self.cls_head = nn.Linear(input_dim, num_classes)
            self.cls_head_0 = nn.Linear(input_dim, 1) # 0类
            self.cls_head_15 = nn.Linear(input_dim, 5) # 1-5类
            self.cls_head_69 = nn.Linear(input_dim, 4) # 6-9类

    def forward(self, x, mask=None):
        # x: [B, N, 256]
        B = x.shape[0]
        
        x = F.normalize(x, p=2, dim=-1)
        #pooling_token = self.pooling.expand(B, -1, -1)
        #x = torch.cat([pooling_token, x], dim=1) # [B, 1+N, 256]

        attn_output, attention_mask = self.attention(x, x, x, key_padding_mask=mask) # self-attention
        
        attn_output = self.ffn(self.norm(attn_output))
        
        attn_output, attention_mask = self.attention2(attn_output, attn_output, attn_output, key_padding_mask=mask)

        attn_output = self.norm2(attn_output)
        
        #patch_feature = attn_output[:, 0, :] # 使用CLS, [B, 256]
        
        patch_feature = self.att_pool(attn_output, mask=mask)

        if self.cls:
            #logit = self.cls_head(patch_feature)
            logit0 = self.cls_head_0(patch_feature) # [N, 1]
            logit15 = self.cls_head_15(patch_feature) # [N, 5]
            logit69 = self.cls_head_69(patch_feature) # [N, 4]

            logit = torch.cat([logit0, logit15, logit69], dim=-1) # [N, 10]

        else:
            logit = None
        
        return patch_feature, logit



class MultiHeadAttentionPool(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionPool, self).__init__()

        self.num_heads = num_heads
        self.attn_mlp = torch.nn.Linear(input_dim, num_heads)
        self.use_lrelu = True

    def forward(self, x, mask=None):
        #batch = torch.zeros(x.shape[0], device=x.device)
        if mask is not None:
            mask = mask.unsqueeze(-1)

        alpha = self.attn_mlp(x)
        
        if self.use_lrelu:
            alpha = F.leaky_relu(alpha)

        if mask is not None:
            # 掩盖掉padding部分
            #alpha = alpha.masked_fill(mask, float('inf'))  # onnx可能不支持
            #alpha = torch.where(mask, torch.full_like(alpha, -1e9), alpha)
            alpha = alpha + mask*(-1e16)
    
        #alpha = softmax(alpha, batch)
        alpha = F.softmax(alpha, dim=1) #[B, N, H]
        out = 0
        #print(alpha.shape, x.shape)
        for head in range(self.num_heads):
            #out += torch.bmm(alpha[:, :, head].unsqueeze(1), x)
            out += alpha[:, :, head].unsqueeze(-1) * x
        #print(out.shape)        
        out = torch.sum(out, dim=1, keepdim=True)
        #print(out.shape) # [1, 1, D]
        #assert 1 == 0
        out = out.squeeze(1)
        #print(out.shape)
        return out #[N_patch, D]

        #out = out.squeeze(1)

        #return out
        #return torch_scatter.scatter_add(out, batch, dim=0)


        #res = torch.zeros((torch.unique(batch).shape[0], out.shape[-1]), dtype=out.dtype).to(out.device)

        #return torch.scatter_add(res, 0, batch.unsqueeze(-1).expand(-1, out.shape[-1]), out)


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super(CustomMultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 创建 Q, K, V 的线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        xavier_normal_(self.q_proj.weight)
        xavier_normal_(self.k_proj.weight)
        xavier_normal_(self.v_proj.weight)

        # 输出线性变换层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.out_proj.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.out_proj.bias, -bound, bound)
 

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 获取 Q, K, V 的投影
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len, embed_dim]

        # 将 Q, K, V 转换为 (batch_size, num_heads, seq_len, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重: 计算 Q 和 K 的点积
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]

        # 缩放因子
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # 应用 attention mask（如果有的话）
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # 应用 padding mask（如果有的话）
        if key_padding_mask is not None:
            #attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn_weights = attn_weights - key_padding_mask.unsqueeze(1).unsqueeze(2)*1e30

        # 归一化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # 计算加权平均（应用注意力权重到值 V）
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并所有头的输出
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)  # [batch_size, seq_len, embed_dim]

        # 输出投影
        output = self.out_proj(attn_output)

        return output, attn_weights
 