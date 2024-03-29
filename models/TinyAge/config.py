from torch.optim import optimizer
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.transforms import Resize, ToTensor


class Celegans_dataset:
    root_dir = 'D:/Dataset/线虫分类数据集'             # 数据集根目录
    test_dir = r'D:\Dataset\线虫分类数据集\分类测试数据集'             # 数据集根目录
    csv_tar_path = '.\csv files'                                # 包含csv文件的文件夹
    csv_file_train = '.\\csv files\cele_df_train.csv'           # 训练集csv文件路径
    csv_file_val = '.\\csv files\cele_df_val.csv'               # 测试集csv文件路径
    csv_file_test = r'D:\Dataset\线虫分类数据集\csv files\cele_df_test.csv'                # 测试集csv文件路径
    # 需要的标签信息,可选的值有:['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
    labels_name_required = 'shoot_days'   
    class_weights_path = 'class_weights.pt'                      

class Config:
    # 数据集相关设置
    model_dir = ''      
    test_dir = Celegans_dataset.test_dir                                        # 使用的模型文件夹
    resume_weights = None                                       # 设置训练起始epoch
    root_dir = Celegans_dataset.root_dir                        
    csv_tar_path = Celegans_dataset.csv_tar_path
    csv_file_train = Celegans_dataset.csv_file_train
    csv_file_val = Celegans_dataset.csv_file_val
    csv_file_test = Celegans_dataset.csv_file_test
    labels_name_required = Celegans_dataset.labels_name_required
    class_weights_path = Celegans_dataset.class_weights_path

    # 训练设置
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = {
        'train_trans': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(0.5, 2)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            normalize
        ]),
        'val_trans': torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            normalize
        ])
    }
    train_batch_size = 128                   # 训练的batch size
    val_batch_size = 8                     # 验证集的batch size
    begin_epoch = 0                         # 起始epoch
    train_require_layers = None   # 需要训练的层
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weight = torch.load('class_weights.pt').to(device)
    loss_function = 'dldl_loss'
    weight = torch.load('class_weights.pt')
    label_type = 'pdf'

    optimizer = 'SGD'
    momentum = 0.9
    dampening = 0
    nesterov = True
    weight_decay = 5E-5

    # 模型相关设置
    TORCH_HOME = 'E:\workspace'         # 设置pytorch路径，用于指定预训练权重下载路径    
    pretrained = True                   # 是否预训练
    pretrained_path = r'E:\\workspace\\python\\线虫实验\\models\\TinyAge\\weights\\tinyage_pretrained.pth'
    learning_rate = 0.001                # 学习率
    num_epochs = 100                  # 迭代次数
    weight_decay = 0.001



config = Config()