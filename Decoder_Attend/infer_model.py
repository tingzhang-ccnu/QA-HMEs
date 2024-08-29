import torch
import torch.nn as nn
import numpy as np
from densenet import DenseNet
from word_embedding import Embedding
from attention_decoder import Attention
from Atten_language import Attented_language


class Inference(nn.Module):
    def __init__(self, params=None):
        super(Inference, self).__init__()
        self.params = params
        self.device = params['device']

        self.encoder = DenseNet(params)

        self.decoder = AttDecoder(params=self.params)

    def forward(self, img, ques, is_train=False):
        visual_feature = self.encoder(img)
        # visual_features [1 dim h w]  probs [1 len voc_size]
        probs, alphas, alphas_l, ques_index = self.decoder(visual_feature, ques, train=is_train)
        return probs, alphas, alphas_l, ques_index


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

        self.gru = nn.GRUCell(self.embedding_dim + self.encoder_out, self.hidden_dim)

        self.attention = Attention(params)
        self.atten_language = Attented_language(params)

        self.embedding_weight = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.hidden_weight = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.context_weight = nn.Linear(self.encoder_out, self.embedding_dim)
        self.atten_l_weight = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_0 = nn.Linear(self.embedding_dim, self.vocab_size)

        self.drop = nn.Dropout(params['decoder']['dropout_p'])

    def forward(self, features, ques, train=False):
        """
        推理阶段 batch_size=1
        features [1 dim h w]
        ques [1 len]
        """

        batch_size, _, height, width = features.shape
        length = ques.size(-1)
        l_mask = torch.ones((batch_size, length)).to(self.device)
        image_mask = torch.ones((batch_size, 1, height, width)).to(self.device)
        alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)

        hidden = self.init_hidden(features, image_mask)  # [B h_dim]
        embedding = self.embedding(torch.ones([batch_size]).long().to(self.device))
        probs = []
        alphas = []
        # 存放关注度最高的ques符号
        alphas_l = []
        ques_index = []

        i = 0
        while i < 200:
            context, alpha, alpha_sum = self.attention(features, hidden, alpha_sum, image_mask)

            hidden = self.gru(torch.cat([embedding, context], dim=1), hidden)  # [B h_dim]
            # ------加入atten_language------
            atten_l, alpha_l = self.atten_language(ques, context)  # [B embed_dim] [B len]

            out = self.hidden_weight(hidden) + self.embedding_weight(embedding) + self.context_weight(context) + \
                  self.atten_l_weight(atten_l)  # [B embed_dim]
            out = self.drop(out)
            prob = self.W_0(out)  # [B vocab_size]
            _, word = prob.max(1)
            embedding = self.embedding(word)
            if word.item() == 0:
                return probs, alphas, alphas_l, ques_index

            """
            alpha_l = np.array(alpha_l.cpu())  # alpha_l [1 len] tensor->numpy
            indx = np.argmax(alpha_l, axis=1)  # 获取维度1上的最大值索引 形如[4]
            alphas_l.append(ques[0][indx[0]])  # 最关注的ques字符(对应的字典索引)
            """

            # """
            indices = torch.max(alpha_l.squeeze(1), 1)[1]  # 获取每个batch中最大值的索引 [B]
            indices = indices.unsqueeze(-1)  # [B 1]
            ques_index.append(int(indices[0].detach().cpu().numpy()))
            word_index = torch.gather(ques, 1, indices)  # [B 1]
            alphas_l.append(int(word_index[0].detach().cpu().numpy()))  # B=1
            # """

            # alphas_l.append(alpha_l)  # alpha_l dot_attention[B 1 len]  attention[B len]

            alphas.append(alpha)
            probs.append(word)
            i += 1
        return probs, alphas, alphas_l, ques_index

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)
