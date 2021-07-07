import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from PIL import Image
import numpy as np

import d2lzh_pytorch as d2l


def get_C_elegants_label(filename, labels_name_required):
    """通过路径文件名获取标签
        
        args:
            filename: eg: '../input/00001/图片整理/10_0.750_0.833/b_1_2_2_1.JPG'
            labels_name_required: str or list in ['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
        return:
            labels:narray类型，[部位，批次，线虫编号，剩余天数，照片编号，拍摄天数]
            部位：1:head, 2: body, 3:tail
            批次：
            编号：
            剩余天数：
            照片编号：
            拍摄天数：缺失为-1
            
    """
    bname = os.path.basename(filename)
    labels_list = bname.split('.')[0].split('_')
    labels_return = []

    if 'part' in labels_name_required:
        part_dic = {'h': 1, 'b': 2, 't': 3}
        labels_return.append(part_dic[labels_list[0]])
    if 'batch' in labels_name_required:
        labels_return.append(labels_list[1])
    if 'elegans_id' in labels_name_required:
        labels_return.append(labels_list[2])
    if 'remaining_days' in labels_name_required:
        labels_return.append(labels_list[3])
    if 'photo_id' in labels_name_required:
        labels_return.append(labels_list[4])
    if 'shoot_days' in labels_name_required:
        if len(labels_list) < 6:
            labels_return.append(-1)
        else:
            labels_return.append(labels_list[5])


    labels = list(map(int, labels_return))
    return np.array(labels) if len(labels) > 1 else labels[0]

def get_images_path_list(DATAPATH):
    """返回所有图片文件绝对路径
        路径为
        /文件夹/图片
        
        args:
            DATAPATH:数据集目录
            
        return:
            图片文件路径
    """
    paths = []
    path_list = os.listdir(DATAPATH)
    for folder in path_list:
        path_folder = os.path.join(DATAPATH, folder)
        if not os.path.isdir(path_folder):
            continue
        files = os.listdir(path_folder)
        for file in files:
            img_path = os.path.join(path_folder, file)
            paths.append(img_path)
    return paths

def get_images_relative_path_list(DATAPATH, skip_missing_shootday=False):
    """返回所有图片文件相对路径
        
        args:
            DATAPATH:数据集目录
            
        return:
            图片文件路径
    """
    paths = []
    path_list = os.listdir(DATAPATH)
    for folder in path_list:
        path_folder = os.path.join(DATAPATH, folder)
        if not os.path.isdir(path_folder):
            continue
        files = os.listdir(path_folder)
        for file in files:
            img_path = os.path.join(folder, file)
            if skip_missing_shootday and (-1 == get_C_elegants_label(img_path, 'shoot_days')):
                continue
            paths.append(img_path)
    return paths

def generate_C_elegans_csv(dataset_path, csv_tar_path, num_train=None, num_val=None, shuffle=False, skip_missing_shootday=False):
    """
        args:
            dataset_path: 数据集路径，例如: 'E:\workspace\线虫数据集\图片整理'
            csv_tar_path: 生成的csv文件保存路径
            num_test: 分配测试集样本数
            shuffle: 是否乱序
            skip_missing_shootday: 是否跳过没有拍摄时间的样本
    """
    columns = ['path']
    image_list = get_images_relative_path_list(dataset_path, skip_missing_shootday)
    if shuffle:
        random.shuffle(image_list)
    if num_train or num_val:
        list_len = len(image_list)
        if num_train == None:
            num_train = list_len - num_val
        if num_val ==None:
            num_val = list_len - num_train
        cele_df_train = DataFrame(image_list[:num_train], columns=columns)
        cele_df_train.to_csv(os.path.join(csv_tar_path, 'cele_df_train.csv'))
        cele_df_val = DataFrame(image_list[num_train:num_val + num_train], columns=columns)
        cele_df_val.to_csv(os.path.join(csv_tar_path, 'cele_df_val.csv'))
    else:
        cele_df = DataFrame(image_list, columns=columns)
        cele_df.to_csv(os.path.join(csv_tar_path, 'cele_df_all.csv'))


class CelegansDataset(Dataset):
    """C.elegans dataset."""

    def __init__(self, labels_name_required, csv_file, root_dir, transform=None):
        """
        Args:
            label: a list or str in ['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_name_required = labels_name_required
        self.cele_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform:
            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.cele_df)

    def __getitem__(self, idx):
        img_path_name = os.path.join(self.root_dir, self.cele_df.iloc[idx, 1])
        img_PIL = Image.open(img_path_name)
        # image = np.array(img_PIL)
        label = get_C_elegants_label(img_path_name, self.labels_name_required) # narray (num_of_labels,)

        image = self.transform(img_PIL)

        sample = {
            'label':label,
            'image':image
        }
        return sample

if __name__ == '__main__':
    # 产生训练集和测试集的csv
    dataset_path = 'E:\workspace\线虫数据集\图片整理'
    csv_tar_path = 'E:\workspace\线虫数据集\图片整理'
    generate_C_elegans_csv(dataset_path, csv_tar_path, num_test=50, shuffle=True)
