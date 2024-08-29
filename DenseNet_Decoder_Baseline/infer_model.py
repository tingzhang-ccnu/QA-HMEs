import torch
import torch.nn as nn

from densenet import DenseNet
from word_embedding import Embedding
from attention_decoder import Attention


class Inference(nn.Module):
    def __init__(self, params=None):
        super(Inference, self).__init__()
        self.params = params
        self.device = params['device']

        self.encoder = DenseNet(params)

        self.decoder = AttDecoder(params=self.params)

    def forward(self, img, is_train=False):
        visual_feature = self.encoder(img)
        probs, alphas = self.decoder(visual_feature, train=is_train)  # features[-1] [B dim h w] probs:[B len voc_size]
        return probs, alphas, visual_feature


class AttDecoder(nn.Module):
    def __init__(self, params=None):
        super(AttDecoder, self).__init__()
        self.params = params
        self.attention_dim = params['decoder']['attention_dim']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['decoder']['hidden_dim']
        self.vocab_size = params['vocab_size']
        self.encoder_out = params['encoder_out']
        self.ratio = params['densenet']['ratio']  # 长宽与原始尺寸缩小比例
        self.device = params['device']

        self.init_weight = nn.Linear(self.encoder_out, self.hidden_dim)

        self.embedding = Embedding(params)
        self.attention = Attention(params)
        self.gru = nn.GRUCell(self.embedding_dim + self.encoder_out, self.hidden_dim)

        self.embedding_weight = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.hidden_weight = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.context_weight = nn.Linear(self.encoder_out, self.embedding_dim)
        self.W_0 = nn.Linear(self.embedding_dim, self.vocab_size)

        self.drop = nn.Dropout(params['decoder']['dropout_p'])

    def forward(self, features, train=False):
        batch_size, _, height, width = features.shape
        image_mask = torch.ones((batch_size, 1, height, width)).to(self.device)
        alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)

        hidden = self.init_hidden(features, image_mask)  # [B h_dim]
        embedding = self.embedding(torch.ones([batch_size]).long().to(self.device))
        probs = []
        alphas = []

        i = 0
        while i < 200:
            context, alpha, alpha_sum = self.attention(features, hidden, alpha_sum, image_mask)
            hidden = self.gru(torch.cat([embedding, context], dim=1), hidden)  # [B h_dim]
            out = self.hidden_weight(hidden) + self.embedding_weight(embedding) + self.context_weight(context)
            out = self.drop(out)
            prob = self.W_0(out)
            _, word = prob.max(1)
            embedding = self.embedding(word)
            if word.item() == 0:
                return probs, alphas
            alphas.append(alpha)
            probs.append(word)
            i += 1
        return probs, alphas

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)
