import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, params):
        super(Embedding, self).__init__()
        self.vocab_size = params['vocab_size']
        self.embedding_dim = params['embedding_dim']
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

    def forward(self, y):
        """
        fusion_model: y [B len]
        attention_decoder: y [B]
        """
        emb = self.embedding(y)
        return emb
