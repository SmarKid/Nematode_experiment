import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import torch
import math
from pandas import DataFrame

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import numpy as np



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

def get_images_relative_path_list(DATAPATH, skip_missing_shootday=True):
    """返回所有图片文件相对路径
        
        args:
            DATAPATH:数据集绝对路径目录
            
        return:
            图片文件路径
    """
    paths = []
    
    # 要加入的文件扩展名
    extense_name = ['.jpg', '.JPG', '.jpeg']
    def search_dir(root_path):
        '''
        遍历文件夹,将文件放入paths中
        '''
        path_list = os.listdir(root_path)
        for f in path_list:
            if os.path.isdir(os.path.join(root_path, f)):
                search_dir(os.path.join(root_path, f))
            elif os.path.isfile(os.path.join(root_path, f)):
                if os.path.splitext(f)[1] not in extense_name:
                    continue
                elif skip_missing_shootday and get_C_elegants_label(f, 'shoot_days') == -1:
                    continue
                else:
                    relpath = os.path.relpath(os.path.join(root_path, f), DATAPATH)
                    paths.append(relpath)
    search_dir(DATAPATH)
    return paths

def generate_C_elegans_csv(dataset_path, csv_tar_path, num_train=None, num_val=None, shuffle=False, skip_missing_shootday=True):
    """
        args:
            dataset_path: 数据集路径，例如: 'E:\workspace\线虫数据集\图片整理'
            csv_tar_path: 生成的csv文件保存路径
            num_train: 分配训练集样本数
            num_val: 分配验证集样本数
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
        cele_df_train = DataFrame(image_list[:num_train], columns=columns, dtype=str)
        if not os.path.exists(csv_tar_path):
            os.mkdir(csv_tar_path)
        cele_df_train.to_csv(os.path.join(csv_tar_path, 'cele_df_train.csv'))
        cele_df_val = DataFrame(image_list[num_train:num_val + num_train], columns=columns)
        cele_df_val.to_csv(os.path.join(csv_tar_path, 'cele_df_val.csv'))
    else:
        cele_df = DataFrame(image_list, columns=columns)
        cele_df.to_csv(os.path.join(csv_tar_path, 'cele_df_all.csv'))


class CelegansDataset(Dataset):
    """C.elegans dataset."""

    def __init__(self, labels_name_required='shoot_days', csv_file=None, root_dir=None,
                 transform=None, label_type='single', fast_load=False, file_path=None):
        """
        Args:
            label: a list or str in ['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_type (string): works only labels_name_required is a str. a str in ['single', 'pdf'], means single class label and transformed pdf label, respectively.
                pdf means probability density function.
            fast_load: load dataset in a faster way
            file_path: compressed dataset file, required when fast_load is True.
        """
        self.transform = transform
        self.label_type = label_type
        self.fast_load = fast_load
        if fast_load:
            images, labels, pdf_labels = torch.load(file_path)
            self.images = images
            self.labels = labels
            self.pdf_labels = pdf_labels
            return
        self.labels_name_required = labels_name_required
        self.cele_df = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        if self.fast_load:
            return len(self.labels)
        return len(self.cele_df)

    def __getitem__(self, idx):
        if self.fast_load:
            label = self.labels[idx]
            image = self.images[idx]
            # show_tensor_img(image, label)
        else:
            img_path_name = os.path.join(self.root_dir, self.cele_df.iloc[idx, 1])
            image = Image.open(img_path_name)
            label = get_C_elegants_label(img_path_name, self.labels_name_required) # narray (num_of_labels,)
        if self.transform:
            image = self.transform(image)
            # show_tensor_img(image, img_path_name)
        sample = {
            'label': label,
            'image': image
        }

        if self.label_type == 'pdf':
            assert isinstance(label, int), 'pdf cannot be applied to multiple labels'
            if self.fast_load:
                pdf_label = self.pdf_labels[idx]
            else:
                pdf_label = [normal_sampling(int(label), i) for i in range(30)]
                pdf_label = [i if i > 1e-15 else 1e-15 for i in pdf_label]
                # size[class_number + 1]
                pdf_label = torch.Tensor(pdf_label)
            sample['pdf_label'] = pdf_label
        
        return sample


def show_tensor_img(tensor_img:torch.Tensor, describe):
    to_pil = transforms.ToPILImage()
    img = tensor_img.cpu().clone()
    img = to_pil(img)
    img.show(title=describe)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)


if __name__ == '__main__':
    # 产生训练集和测试集的csv
    dataset_path = r'E:\workspace\线虫数据集\分类数据集'
    csv_tar_path = r'E:\workspace\线虫数据集\分类数据集'
    generate_C_elegans_csv(dataset_path, csv_tar_path, num_val=600, shuffle=True)
