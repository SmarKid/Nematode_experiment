import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from lib.celegans_dataset import CelegansDataset, get_C_elegants_label
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(args, config, network):
    net = network()
    net.to(device)
    net.eval() # 评估模式, 这会关闭dropout
    if args.resume_weights:
        model_file = os.path.join("./models/", args.model_dir, 'weights/model-%d.pth' % args.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        net.load_state_dict(check_point)
    ex_name = os.path.splitext(args.file_path)[1]
    if ex_name in ['.JPG', '.jpeg', '.jpg']:
        file_path = os.path.normpath(args.file_path)
        img_PIL = Image.open(file_path)
        label = get_C_elegants_label(args.file_path, config.labels_name_required)
        image = config.trans['val_trans'](img_PIL)
        image = torch.unsqueeze(image, 0)

        output = net(image.to(device))
        plt.plot(output.detach().cpu().numpy().reshape(-1, 1))
        plt.savefig('output.jpg')
        print('output:', output)
        # infer_label = output.argmax(axis=1).item()
         
        rank = torch.Tensor([i for i in range(30)]).to(device)
        infer_label = torch.sum(output * rank, dim=1)
        print('预测标签为: %lf' % infer_label)
        print('真实标签为: %d' % label)
        # print('预测置信度为: %lf' % output[0, infer_label])
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None,required=True, type=int)  
    parser.add_argument('--file_path', '-f', default=None, required=True, type=str)  
    args = parser.parse_args()
    model_root_dir = os.path.join('./models/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    from network import network
    inference(args, config, network)

