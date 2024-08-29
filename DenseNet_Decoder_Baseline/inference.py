import os
import json
import cv2
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
from utils import load_config, compute_edit_distance, draw_vision_attention_map
from infer_model import Inference
from dataset import Words


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='./config.yaml', help='config file')
    parser.add_argument('--device', default='cpu', help='device')

    parser.add_argument('--dictionary', default='dictionary.txt', help='dictionary file')
    parser.add_argument('--test_image_path', default='test_image', help='')
    parser.add_argument('--test_label_path', default='test.txt', help='')
    parser.add_argument('--checkpoint', default='last-E200_WordRate0.9783_ExpRate0.5928.pth', help='the path of checkpoint')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    params = load_config(args.config)
    params['device'] = args.device

    dic = Words(args.dictionary)
    params['dic_num'] = len(dic)

    model = Inference(params)
    model = model.to(args.device)

    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()

    with open(args.test_label_path, 'r') as f:
        lines = f.readlines()

    line_right = 0
    e1, e2, e3 = 0, 0, 0
    right_case, e1_case, e2_case, e3_case = {}, {}, {}, {}
    bad_case = {}
    total = {}

    count = 0

    with torch.no_grad():
        for line in tqdm(lines):
            # image_name, label = line.strip().split('\t')
            ques, image_name, label = line.strip().split('\t')
            # input_ques = ques.split(' ')
            name = image_name.split('.')[0]
            input_label = label.split(' ')

            image = Image.open(os.path.join(args.test_image_path, image_name))
            image = F.to_tensor(image.convert("L"))
            image = image.unsqueeze(0).to(args.device)

            input_label = dic.encode(input_label)
            input_label = torch.LongTensor(input_label)
            input_label = input_label.unsqueeze(0).to(args.device)

            # input_ques = dic.encode(input_ques)
            # input_ques = torch.LongTensor(input_ques)
            # input_ques = input_ques.unsqueeze(0).to(args.device)

            probs, alphas, feature = model(image)  # attention map not add

            prediction = dic.decode(probs)

            if prediction == label:  # ExpRate
                line_right += 1

            distance = compute_edit_distance(prediction, label)
            # ≤1 ≤2 ≤3
            if distance <= 1:
                e1 += 1
            if distance <= 2:
                e2 += 1
            if distance <= 3:
                e3 += 1

            # total[image_name] = {
            #     'label': label,
            #     'predL': prediction,
            #     'flag': distance
            # }

    print('ExpRate:{}'.format(line_right / len(lines)))
    print('≤1:{}'.format(e1 / len(lines)))
    print('≤2:{}'.format(e2 / len(lines)))
    print('≤3:{}'.format(e3 / len(lines)))
    print(len(total))

    # with open("./example/total.json", "w", encoding='utf-8') as fb:
    #     json.dump(total, fb, indent=2, ensure_ascii=False)

