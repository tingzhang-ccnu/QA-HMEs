import torch
import torch.nn as nn
from densenet import DenseNet
from fusion_model import Fusion
from attention_decoder import AttDecoder


class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.params = params
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params)
        self.fusion = Fusion(params)
        self.decoder = AttDecoder(params)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, ques, l_mask, img, img_mask, label, label_mask, is_train=True):
        visual_feature = self.encoder(img)
        fusion_feature = self.fusion(visual_feature, ques, l_mask)
        probs, alphas = self.decoder(fusion_feature, label, img_mask, label_mask, train=is_train)
        loss = self.loss_fn(probs.contiguous().view(-1, probs.shape[-1]), label.view(-1))
        average_loss = (loss * label_mask.view(-1)).sum() / (
                label_mask.sum() + 1e-10) if self.use_label_mask else loss
        return probs, average_loss
