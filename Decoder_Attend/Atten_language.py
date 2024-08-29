import torch
import torch.nn as nn
from word_embedding import Embedding

"""
解码器新增注意力模块
query:每一步注意后的视觉向量 context[B D]
key value:语言嵌入向量
    language:[batch len] --> embedding:[batch len embed_dim]
"""


class Attented_language(nn.Module):
    def __init__(self, params):
        super(Attented_language, self).__init__()
        self.params = params
        self.query_dim = params['encoder_out']
        self.embedding_dim = params['embedding_dim']
        self.attention_dim = params['decoder']['attention_dim']
        # self.D = params['encoder_out']

        self.embedding = Embedding(params)
        self.W_q = nn.Linear(self.query_dim, self.attention_dim)
        self.W_l = nn.Linear(self.embedding_dim, self.attention_dim)

        self.V = nn.Linear(self.attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, language, context):
        """
        context:[B D]
        language:[batch len] --> embedding:[batch len embed_dim]
        """

        # attention
        embed_l = self.embedding(language)  # [B len]->[B len embed_dim]
        query = self.W_q(context)  # [B D]->[B attention_dim]
        W_language = self.W_l(embed_l)  # [B len embed_dim]->[B len attention_dim]
        attention_score = torch.tanh(W_language + query[:, None, :])  # [B len attention_dim]
        energy = (self.V(attention_score)).squeeze(-1)  # [B len attention_dim]->[B len 1]->[B len]
        alpha = self.softmax(energy)  # [B len]
        atten_l = (alpha.unsqueeze(2) * embed_l).sum(dim=1)  # [B len embed_dim]->[B embed_dim]
        return atten_l, alpha


