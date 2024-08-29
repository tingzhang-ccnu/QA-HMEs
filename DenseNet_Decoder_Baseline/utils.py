import torch
import os
import cv2
import math
import yaml
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('UTF-8 encoding....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    return params


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def save_checkpoint(filename, model, optimizer, epoch):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, filename)
    return


def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)

    # SequenceMatcher: isjunk是否跳过的值 默认为None
    # ratio()函数计算序列a和b的相似度, ratio=2*M/T M为匹配的字符数, T为两个序列的总字符数
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (
                len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(),
                                         mask.cpu().detach().numpy())]

    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


# 计算余弦相似度
def cos_sim(vec1, vec2):
    vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)  # vec1 / |vec1|
    vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)  # vec2 / |vec2|
    return vec1 @ vec2.permute(0, 2, 1)


def draw_vision_attention_map(image, attention):
    h, w = image.shape
    attention = cv2.resize(attention, (w, h))
    attention_heatmap = ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))*255).astype(np.uint8)
    attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_OCEAN)  # cv2.COLORMAP_HSV TWILIGHT INFERNO
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    attention_map = cv2.addWeighted(attention_heatmap, 0.4, image_new, 0.6, 0.)
    return attention_map
