# 仅用image, label

import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np


class MyDataset(Dataset):
    def __init__(self, image_path, label_path, dic, train=True):
        self.images_path = image_path
        self.dic = dic
        self.train = train

        self.ques = []
        self.image = []
        self.label = []

        count = 0
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                self.ques.append(parts[0].split(' '))
                self.image.append(parts[1])  # 图像名
                self.label.append(parts[2].split(' '))
                count = count + 1
        self.len = count

    def __len__(self):  # 数据集大小
        return self.len

    def __getitem__(self, index):  # 根据索引index获取数据信息
        image_name = self.image[index]  # 图像名
        img_path = os.path.join(self.images_path, image_name)  # 图像路径
        img = Image.open(img_path).convert("L")  # 读取图像, RGB->Gray
        img = F.to_tensor(img)  # [1, H, W]: (img/255)

        # ques = self.ques[index]
        # ques.append('eos')
        # ques = self.dic.encode(ques)
        # ques = torch.LongTensor(ques)

        label = self.label[index]
        label.append('eos')
        label = self.dic.encode(label)
        label = torch.LongTensor(label)

        return img, label


def get_dataset(args):
    dic = Words(args.dictionary)

    train_dataset = MyDataset(args.train_image_path, args.train_label_path, dic, train=True)
    eval_dataset = MyDataset(args.eval_image_path, args.eval_label_path, dic, train=False)
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, eval_loader


def collate_fn(batch_images):
    # batch_images: 是一个列表, 大小为batch_size, 元素为[ques, img, label] (img[C H W])
    max_width, max_height, max_ques, max_length = 0, 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        # max_ques = item[0].shape[0] if item[0].shape[0] > max_ques else max_ques
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))
    # ques, ques_masks = torch.zeros((len(proper_items), max_ques)).long(), torch.zeros((len(proper_items), max_ques))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
        # q = proper_items[i][0].shape[0]
        # ques[i][:q] = proper_items[i][0]
        # ques_masks[i][:q] = 1
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, dict_path):
        with open(dict_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        # 'sep'.join(seq) 以sep作为分隔符, 将seq所有的元素合并成一个新的字符串
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

