import torch
import torch.nn as nn
import torch.nn.functional as F
from word_embedding import Embedding

"""
DenseNet->visual feature:[B C H W]  C=684
text(ques) feature: [B len]
Dot-Product Attention: 
    query: visual feature 
    key,value: text feature
"""


class Fusion(nn.Module):
    def __init__(self, params):
        super(Fusion, self).__init__()
        self.params = params
        self.v_in_channel = params['encoder_out']
        self.out_channel = params['fusion']['out_channel']  # 684
        self.dropout = params['fusion']['f_dropout']
        self.embedding = Embedding(params)
        self.W_m = nn.Sequential(nn.Conv1d(self.v_in_channel, self.out_channel, 1, 1),  # the init function sets bias to 0 if bias is True
                                 nn.GELU(),
                                 nn.Dropout(self.dropout)
                                 )
        self.DP_attention = ScaledDotProductAttention(params)

        self.W_o = nn.Sequential(nn.Conv1d(self.out_channel, self.out_channel, 1, 1),
                                 nn.GELU(),
                                 nn.Dropout(self.dropout)
                                 )

        # language gate
        self.res_gate = nn.Sequential(
            nn.Linear(self.out_channel, self.out_channel, bias=False),
            nn.ReLU(),
            nn.Linear(self.out_channel, self.out_channel, bias=False),
            nn.Tanh()
        )

    def forward(self, x, l, l_mask):
        """
        x: visual feature, [B C H W]  (C=v_in_channel=out_channel)
        l: ques, [B len]  embedding->[B len l_in_channel] (l_in_channel=embedding_dim)
        l_mask: [B len]
        """
        batch, channel, H, W = x.shape
        vis = self.W_m(x.view(batch, channel, -1))  # x: [B C H W]->[B C H*W]
        l_in_fea = self.embedding(l).permute(0, 2, 1)  # [B len]->[B len l_in_channel]->[B l_in_channel len]
        l_fea = self.DP_attention(x, l_in_fea, l_mask)  # [B H*W out_channel]

        g = x.view(batch, -1, channel) + self.res_gate(l_fea) * l_fea  # apply Gate [B H*W out_channel]
        g = g.view(batch, -1, H, W)  # [B out_channel H W]

        return g


class ScaledDotProductAttention(nn.Module):
    def __init__(self, params):
        super(ScaledDotProductAttention, self).__init__()
        self.params = params
        self.v_in_channels = params['encoder_out']
        self.l_in_channels = params['embedding_dim']
        self.out_channels = params['fusion']['out_channel']
        self.num_heads = params['fusion']['num_heads']
        self.l_k = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.out_channels, kernel_size=1, stride=1),
        )
        self.l_v = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.out_channels, kernel_size=1, stride=1),
        )
        self.v_q = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )
        self.W = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        """
        x: [B C H W]
        l: [B l_in_channel len]
        l_mask: [B len]
        """
        batch, channel, H, W = x.shape
        x = x.view(batch, channel, -1)  # [B C H*W]
        l_mask = l_mask.unsqueeze(1)  # [B len]->[B 1 len]
        query = self.v_q(x)  # [B out_channel H*W]
        query = query.permute(0, 2, 1)  # [B H*W out_channel]
        key = self.l_k(l)  # [B out_channel len]
        value = self.l_v(l)  # [B out_channel len]
        key = key * l_mask  # [B out_channel len]
        value = value * l_mask  # [B out_channel len]
        len = value.size(-1)
        # [B H*W num_heads out_channels/num_heads]->[B num_heads H*W out_channels/num_heads]
        query = query.reshape(batch, H*W, self.num_heads, self.out_channels // self.num_heads).permute(0, 2, 1, 3)
        # [B num_heads out_channels/num_heads len]
        key = key.reshape(batch, self.num_heads, self.out_channels // self.num_heads, len)
        value = value.reshape(batch, self.num_heads, self.out_channels // self.num_heads, len)
        l_mask = l_mask.unsqueeze(1)  # [B 1 1 len]

        sim_map = torch.matmul(query, key)  # 矩阵乘积qk [B num_heads H*W len]
        sim_map = (self.out_channels ** -.5) * sim_map  # scaled dot product  1/sqrt{out_channel} * qk

        sim_map = sim_map + (1e4 * l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # [B num_heads H*W len]
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # [B num_heads H*W out_channel/num_head]
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch, H*W, self.out_channels)  # [B H*W out_channel]
        out = out.permute(0, 2, 1)  # [B out_channel H*W]
        out = self.W(out)  # [B out_channel H*W]
        out = out.permute(0, 2, 1)  # [B H*W out_channel]

        return out
