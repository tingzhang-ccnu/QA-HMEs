import os
import torch
import cv2
import json
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
from utils import load_config, compute_edit_distance
# from utils import draw_vision_attention_map, draw_language_attention_map
# from Process.drawmap import draw_vision_attention_map, draw_language_attention_map
from infer_model import Inference
from dataset import Words


def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', default='./config.yaml', help='config file')
    parser.add_argument('--device', default='cpu', help='device')

    parser.add_argument('--dictionary', default='dictionary.txt', help='dictionary file')
    parser.add_argument('--test_image_path', default='test_image', help='')
    parser.add_argument('--test_label_path', default='test.txt', help='')  # no_repeat_
    parser.add_argument('--checkpoint', default='last-E200_WordRate0.9872_ExpRate0.5991.pth', help='the path of checkpoint')
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

    state = torch.load(args.checkpoint, map_location='cpu')  # 加载网络
    model.load_state_dict(state['model'])  # 加载网络参数
    model.eval()

    with open(args.test_label_path, 'r') as f:
        lines = f.readlines()

    line_right = 0
    e1, e2, e3 = 0, 0, 0
    right_case, e1_case, e2_case, e3_case = {}, {}, {}, {}
    bad_case = {}
    total = {}
    case = {}

    with torch.no_grad():
        for line in tqdm(lines):

            ques, image_name, label = line.strip().split('\t')
            name = image_name.split('.')[0]
            input_ques = ques.split(' ')
            input_label = label.split(' ')

            image = Image.open(os.path.join(args.test_image_path, image_name))
            image = F.to_tensor(image.convert("L"))
            image = image.unsqueeze(0).to(args.device)  #

            input_label = dic.encode(input_label)
            input_label = torch.LongTensor(input_label)
            input_label = input_label.unsqueeze(0).to(args.device)

            input_ques = dic.encode(input_ques)
            input_ques = torch.LongTensor(input_ques)
            input_ques = input_ques.unsqueeze(0).to(args.device)  # [1 len]

            # alphas_l[B len]
            probs, alphas_v, alphas_l, ques_index = model(image, input_ques)  # attention map not add

            # ----------------draw attention map------------------
            # if not os.path.exists(os.path.join(params['vision_attention_map_path'], name)):
            #     os.makedirs(os.path.join(params['vision_attention_map_path'], name), exist_ok=True)
            # if not os.path.exists(os.path.join(params['language_attention_map_path'], name)):
            #     os.makedirs(os.path.join(params['language_attention_map_path'], name), exist_ok=True)
            # for i in range(image.shape[0]):
            #     # image: [1 1 H W]
            #     # 阻断反向传播->移至CPU->转为numpy数据
            #     img = image[i][0].detach().cpu().numpy() * 255   # [H W] * 255
            #     # image_attention_map
            #     for step in range(len(probs)):  # probs [nsteps]
            #         word_atten = alphas_v[step][0].detach().cpu().numpy()  # [H W]
            #         word_heatmap = draw_vision_attention_map(img, word_atten)
            #         cv2.imwrite(os.path.join(params['vision_attention_map_path'], name, f'word_{step}.jpg'), word_heatmap)
            #     # language_attention_map
            # for step in range(len(probs)):
            #     # print(alphas_l[step])
            #     lg_atten = alphas_l[step][0].unsqueeze(0).detach().cpu().numpy()  # [1 ques_len]  加性注意力加.unsqueeze(0)
            #     path = os.path.join(params['language_attention_map_path'], name)
            #     out = dic.decode(probs[step])
            #     draw_language_attention_map(lg_atten, ques, path, step, out)

            prediction = dic.decode(probs)

            attened_language = dic.decode(alphas_l)
            ques_index = ' '.join([str(item) for item in ques_index])

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

            """
            total[image_name] = {
                'ques': ques,
                'label': label,
                'predL': prediction,
                'atten': attened_language,
                'Qindx': ques_index,
                'flag': distance
            }
            """

    print('ExpRate:{}'.format(line_right / len(lines)))
    print('≤1:{}'.format(e1 / len(lines)))
    print('≤2:{}'.format(e2 / len(lines)))
    print('≤3:{}'.format(e3 / len(lines)))

# with open("./example/total.json", "w", encoding='utf-8') as fr:
#     json.dump(total, fr, indent=2, ensure_ascii=False)
