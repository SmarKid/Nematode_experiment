import torchvision
import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize, ToTensor


class Celegans_dataset:
    root_dir = 'D:/Dataset/线虫分类数据集'  # 数据集根目录
    test_dir = r'D:\Dataset\线虫分类数据集\分类测试数据集'  # 测试数据集根目录
    csv_tar_path = './csv files'  # 包含csv文件的文件夹
    csv_file_train = './csv files/cele_df_train.csv'  # 训练集csv文件路径
    csv_file_val = './csv files/cele_df_val.csv'  # 测试集csv文件路径
    csv_file_test = r'D:\Dataset\线虫分类数据集\csv files\cele_df_test.csv'  # 测试集csv文件路径
    # 需要的标签信息,可选的值有:['part', 'batch', 'elegans_id', 'remaining_days', 'photo_id', 'shoot_days']
    labels_name_required = 'shoot_days'
    class_weights_path = 'class_weights.pt'

class Config:
    # 数据集相关设置
    model_dir = ''                                              # 使用的模型文件夹
    resume_weights = None                                       # 设置训练起始epoch
    root_dir = Celegans_dataset.root_dir                        
    csv_tar_path = Celegans_dataset.csv_tar_path
    csv_file_train = Celegans_dataset.csv_file_train
    csv_file_val = Celegans_dataset.csv_file_val
    labels_name_required = Celegans_dataset.labels_name_required
    class_weights_path = Celegans_dataset.class_weights_path
    test_dir = Celegans_dataset.test_dir                                        # 使用的模型文件夹

    # 训练设置
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = {
        'train_trans': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((300, 300), scale=(0.5, 1), ratio=(0.5, 2)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            normalize
        ]),
        'val_trans': torchvision.transforms.Compose([
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            normalize
        ])
    }
    loss_function = 'dldl_loss'
    lambda1 = 0
    weight = torch.load('class_weights.pt')
    train_batch_size = 16                   # 训练的batch size
    val_batch_size = 1                     # 验证集的batch size
    begin_epoch = 0                         # 起始epoch
    train_require_layers = None   # 需要训练的层
    # pretrained_path = 'D:\\WorkSpace\\PycharmProjects\\Nematode_experiment\\models\\efficientnetV2\\weights\\pre_efficientnetv2-s.pth' # 预训练权重路径
    pretrained_path = "E:\workspace\python\线虫实验\models\dl_efficientnetV2\weights\pre_efficientnetv2-s.pth"     # 预训练权重路径
    # 模型相关设置
    learning_rate = 0.001                # 学习率
    num_epochs = 100               # 迭代次数

    # 训练设置
    optimizer = 'SGD'
    optim_args = {
        'AdamW': {
            'lr': 0.0000005,  # 学习率
            'weight_decay': 5E-2,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        'SGD': {
            'lr': 0.001,  # 学习率
            'weight_decay': 5E-5,
            'nesterov': False,
            'dampening': 0,
            'momentum': 0.9
        }
    }


config = Config()