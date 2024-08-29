import os
import random
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from dataset import get_dataset
from encoder_decoder import Encoder_Decoder
from tqdm import tqdm
from utils import load_config, cal_score, Meter, save_checkpoint, update_lr
import argparse


def train(args, model, optimizer, epoch, train_loader, writer=None):
    model.train()
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for step, data in enumerate(pbar):
            # ques: [B q_len]  l_msk: [B q_len]  img: [B C H W]  img_mask: [B 1 H W]
            # label: [B len] label_mask: [B len]
            ques, l_mask, img, img_mask, label, label_mask = data

            ques, l_mask = ques.to(args.device), l_mask.to(args.device)
            label, label_mask = label.to(args.device), label_mask.to(args.device)
            img, img_mask = img.to(args.device), img_mask.to(args.device)

            batch, time = label.shape[:2]
            if args.lr_decay == 'cosine':
                update_lr(optimizer, epoch, step, len(train_loader), args.epochs, args.lr)
            optimizer.zero_grad()
            # probs:[B len voc_size]  alphas:[B len h w]
            probs, loss = model(ques, l_mask, img, img_mask, label, label_mask)

            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient)
            optimizer.step()

            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, label, label_mask)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            total_train_step = epoch * len(train_loader) + step + 1
            if writer:
                writer.add_graph(model, ques, l_mask, img, img_mask, label, label_mask)
                # print("train_step:{}, loss:{}, WRate:{:.4f}, ERate:{:.4f}".format(total_train_step, loss.item(),
                #                                                            word_right / length, exp_right / cal_num))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                writer.add_scalar('train_WordRate', wordRate, total_train_step)
                writer.add_scalar('train_ExpRate', ExpRate, total_train_step)
            pbar.set_description(f'{epoch + 1} train_step:{total_train_step} loss:{loss.item():.4f} '
                                 f'WRate:{word_right / length:.4f} ERate:{exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/train_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, exp_right / cal_num


def eval(args, model, epoch, test_loader, writer=None):
    # evaluate
    model.eval()
    loss_meter = Meter()
    word_right, exp_right, length, cal_num = 0, 0, 0, 0

    with tqdm(test_loader, total=len(test_loader)) as pbar, torch.no_grad():
        for step, data in enumerate(pbar):
            ques, l_mask, img, img_mask, label, label_mask = data
            ques, l_mask = ques.to(args.device), l_mask.to(args.device)

            label, label_mask = label.to(args.device), label_mask.to(args.device)
            img, img_mask = img.to(args.device), img_mask.to(args.device)
            batch, time = label.shape[:2]

            probs, loss = model(ques, l_mask, img, img_mask, label, label_mask, is_train=False)
            loss_meter.add(loss.item())

            wordRate, ExpRate = cal_score(probs, label, label_mask)
            word_right = word_right + wordRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch
            total_eval_step = epoch * len(test_loader) + step + 1
            if writer:
                writer.add_scalar("eval_loss", loss.item(), total_eval_step)
                writer.add_scalar('eval_WordRate', wordRate, total_eval_step)
                writer.add_scalar('eval_ExpRate', ExpRate, total_eval_step)
            pbar.set_description(f'{epoch + 1} eval_step:{total_eval_step} loss:{loss.item():.4f} '
                                 f'WRate:{word_right / length:.4f} ERate:{exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, exp_right / cal_num


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='./config.yaml', help='config file')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')

    parser.add_argument('--dictionary', default='dictionary.txt', help='dictionary file')
    parser.add_argument('--train_image_path', default='train_image', help='')
    parser.add_argument('--train_label_path', default='train.txt', help='')
    parser.add_argument('--eval_image_path', default='eval_image', help='')
    parser.add_argument('--eval_label_path', default='eval.txt', help='')

    parser.add_argument('--log_dir', default='./logs', help='')
    parser.add_argument('--saved_dir', default='./models', help='save dir of trained models')
    parser.add_argument('--resume', default=False, help='resume from checkpoint: True or False')
    parser.add_argument('--checkpoint', default='', help='the path of checkpoint')

    parser.add_argument('--lr', default=1, type=float, help='ABM:0.001')
    parser.add_argument('--eps', default=1e-6, type=float, help='ABM:1e-6')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='ABM:1e-4')
    parser.add_argument('--lr_decay', default='cosine', help='')

    parser.add_argument('--gradient_clip', default=True, help='')
    parser.add_argument('--gradient', default=100, help='')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    params = load_config(args.config)
    params['device'] = args.device

    train_loader, test_loader = get_dataset(args)  # 加载数据

    model = Encoder_Decoder(params)
    model = model.to(args.device)

    # 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # decoder优化器
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.weight_decay)  # ABM

    if not os.path.exists(args.saved_dir):
        os.mkdir(args.saved_dir)

    if args.resume:
        state = torch.load(args.checkpoint)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch'] + 1
        min_exprate = float(args.checkpoint.split('/')[-1].split('_')[-1][7:13])
        print("Successfully load model from epoch {}".format(start_epoch))
    else:
        start_epoch = 0
        min_exprate = 0

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        # print("Epoch:{}".format(epoch+1))
        train_loss, train_word_score, train_exprate = train(args, model, optimizer, epoch, train_loader, writer=writer)
        # eval_loss: 平均损失
        eval_loss, eval_word_score, eval_exprate = eval(args, model, epoch, test_loader, writer=writer)
        print(f'Epoch: {epoch + 1} train_loss: {train_loss:.4f} word score: {train_word_score:.4f} ExpRate: {train_exprate:.4f}')
        print(f'Epoch: {epoch+1} eval_loss: {eval_loss:.4f} word score: {eval_word_score:.4f} ExpRate: {eval_exprate:.4f}')

        # save the last model
        if (epoch+1) % 25 == 0:  # 每10个epoch保存一次
            last_path = os.path.join(args.saved_dir, 'last')
            if not os.path.exists(last_path):
                os.mkdir(last_path)
            last_model_name = f'last-E{epoch+1}_WordRate{eval_word_score:.4f}_ExpRate{eval_exprate:.4f}.pth'
            save_checkpoint(os.path.join(last_path, last_model_name), model, optimizer, epoch)
            print("Saving the last model of epoch {}...".format(epoch + 1))

        # save the best model
        if eval_exprate > min_exprate:
            min_exprate = eval_exprate
            best_path = os.path.join(args.saved_dir, 'best')
            if not os.path.exists(best_path):
                os.mkdir(best_path)
            best_model_name = f'best-E{epoch+1}_WordRate{eval_word_score:.4f}_ExpRate{eval_exprate:.4f}.pth'
            save_checkpoint(os.path.join(best_path, best_model_name), model, optimizer, epoch)
            print("Saving the best model of epoch {}...".format(epoch+1))

    writer.close()

