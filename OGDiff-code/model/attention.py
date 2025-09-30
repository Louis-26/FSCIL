# 导入必要的库
import torch
from torch import nn
from torch.nn import functional as F
import math

# 自注意力机制实现
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # 输入投影层，同时生成Q、K、V
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # 输出投影层
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        # 注意力头数
        self.n_heads = n_heads
        # 每个头的维度
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # 保存输入张量的形状
        input_shape = x.shape
        # 获取批量大小、序列长度和嵌入维度
        batch_size, sequence_length, d_embed = input_shape
        # 中间形状，用于多头注意力计算
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # 将输入投影为Q、K、V并分成三部分
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # 重塑Q、K、V的形状并转置以便多头计算
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # 计算注意力权重 (Q @ K^T)
        weight = q @ k.transpose(-1, -2)
        # 如果使用因果掩码，屏蔽未来信息
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        # 缩放注意力权重
        weight /= math.sqrt(self.d_head)
        # 应用softmax归一化
        weight = F.softmax(weight, dim=-1)

        # 计算注意力输出 (权重 @ V)
        output = weight @ v
        # 转置并重塑回原始形状
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        # 通过输出投影层
        output = self.out_proj(output)
        return output

# 交叉注意力机制实现
class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Q投影层 (来自x)
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # K投影层 (来自y)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # V投影层 (来自y)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # 输出投影层
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        # 注意力头数
        self.n_heads = n_heads
        # 每个头的维度
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # 保存输入张量的形状
        input_shape = x.shape
        # 获取批量大小、序列长度和嵌入维度
        batch_size, sequence_length, d_embed = input_shape
        # 中间形状，用于多头注意力计算
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # 分别投影Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # 重塑Q、K、V的形状并转置以便多头计算
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # 计算注意力权重 (Q @ K^T)
        weight = q @ k.transpose(-1, -2)
        # 缩放注意力权重
        weight /= math.sqrt(self.d_head)
        # 应用softmax归一化
        weight = F.softmax(weight, dim=-1)

        # 计算注意力输出 (权重 @ V)
        output = weight @ v
        # 转置并重塑回原始形状
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        # 通过输出投影层
        output = self.out_proj(output)
        return output