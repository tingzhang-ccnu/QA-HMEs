import torch
import torch.nn as nn
from word_embedding import Embedding


class Attention(nn.Module):
    def __init__(self, params):
        super(Attention, self).__init__()
        self.params = params
        self.hidden_dim = params['decoder']['hidden_dim']
        self.attention_dim = params['decoder']['attention_dim']
        self.D = params['encoder_out']

        self.conv_Q = nn.Conv2d(1, 512, kernel_size=11, padding=5, bias=False)

        self.U_a = nn.Conv2d(self.D, self.attention_dim, kernel_size=1)
        self.W_a = nn.Linear(self.hidden_dim, self.attention_dim)  # ht-1
        self.U_f = nn.Linear(512, self.attention_dim)

        self.V_a = nn.Linear(self.attention_dim, 1)

    def forward(self, features, hidden, alpha_sum, image_mask=None):
        Ua_fea = self.U_a(features)  # [B D H W]->[B attention_dim H W]
        Wa_h = self.W_a(hidden)  # [B h_dim]->[B attention_dim]
        cover_F = self.conv_Q(alpha_sum)  # [B 1 H W]->[B 512 H W]
        cover_vector = self.U_f(cover_F.permute(0, 2, 3, 1))  # permute [B H W 512]->[B H W attention_dim]

        attention_score = torch.tanh(Ua_fea.permute(0, 2, 3, 1) + Wa_h[:, None, None, :] + cover_vector)  # [B H W attention_dim]

        energy = self.V_a(attention_score)  # [B H W 1]
        energy_exp = torch.exp(energy.squeeze(-1))  # [B H W]
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:, None, None])  # [B H W]
        alpha_sum = alpha[:, None, :, :] + alpha_sum  # [B 1 H W]
        context = (alpha[:, None, :, :] * features).sum(-1).sum(-1)  # [B D]
        return context, alpha, alpha_sum


class AttDecoder(nn.Module):
    def __init__(self, params):
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

        self.embedding_weight = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.hidden_weight = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.context_weight = nn.Linear(self.encoder_out, self.embedding_dim)
        self.W_0 = nn.Linear(self.embedding_dim, self.vocab_size)

        self.drop = nn.Dropout(params['decoder']['dropout_p'])

    def forward(self, features, labels, images_mask, label_mask, train=True):
        batch_size, n_step = labels.shape
        height, width = features.shape[2:]
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        probs = torch.zeros((batch_size, n_step, self.vocab_size)).to(device=self.device)
        alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        alphas = torch.zeros((batch_size, n_step, height, width)).to(device=self.device)

        hidden = self.init_hidden(features, images_mask)  # [B h_dim]

        if train:
            for i in range(n_step):
                embedding = self.embedding(labels[:, i-1]) if i else self.embedding(torch.ones([batch_size]).long().to(self.device))  # [B embed_dim]
                context, alpha, alpha_sum = self.attention(features, hidden, alpha_sum, images_mask)
                hidden = self.gru(torch.cat([embedding, context], dim=1), hidden)  # [B h_dim]

                out = self.hidden_weight(hidden) + self.embedding_weight(embedding) + self.context_weight(context)  # [B embed_dim]
                if self.params['decoder']['dropout']:
                    out = self.drop(out)  # [B embed_dim]
                prob = self.W_0(out)  # [B vocab_size]
                probs[:, i] = prob
                alphas[:, i] = alpha
        else:
            embedding = self.embedding(torch.ones([batch_size]).long().to(self.device))
            for i in range(n_step):
                context, alpha, alpha_sum = self.attention(features, hidden, alpha_sum, images_mask)
                hidden = self.gru(torch.cat([embedding, context], dim=1), hidden)  # [B h_dim]

                out = self.hidden_weight(hidden) + self.embedding_weight(embedding) + self.context_weight(context)
                if self.params['decoder']['dropout']:
                    out = self.drop(out)
                prob = self.W_0(out)
                _, word = prob.max(1)
                embedding = self.embedding(word)
                probs[:, i] = prob
                alphas[:, i] = alpha
        return probs, alphas

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)

