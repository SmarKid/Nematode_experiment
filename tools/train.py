from lib.make_optimizer import make_optimizer
import torch
import logging
import argparse
import sys
import os
import tqdm
sys.path.append('../')
from lib.celegans_dataset import CelegansDataset
from lib.utils import Trainer
from lib.utils import Evaluator
from torch.utils.tensorboard import SummaryWriter


def set_params():
    # 配置日志
    file_handler = logging.FileHandler(filename='./log/logging.log', mode='a', encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stderr)
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %p',
        handlers=[file_handler, stream_handler],
        level=logging.INFO
    )
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, type=int)
    parser.add_argument('--file_path', '-f', default=None, type=str)
    args = parser.parse_args()

    # 从模型文件夹导入network和config
    model_root_dir = os.path.join('./models/', args.model_dir)
    sys.path.insert(0, model_root_dir)
    from config import config
    config.model_dir = args.model_dir
    config.resume_weights = args.resume_weights
    from network import network
    return config, network


def get_dataloader(config):
    train_file_path = 'D:\\Dataset\\线虫分类数据集transformed\\C_train_dataset_transformed.pth'
    train_set = CelegansDataset(transform=config.trans['train_trans'],
                                label_type='pdf',
                                fast_load=True,
                                root_dir=config.root_dir,
                                csv_file=config.csv_file_train,
                                file_path=train_file_path)
    val_file_path = 'D:\\Dataset\\线虫分类数据集transformed\\C_val_dataset_transformed.pth'
    val_set = CelegansDataset(transform=config.trans['val_trans'],
                              label_type='pdf',
                              fast_load=True,
                              root_dir=config.root_dir,
                              csv_file=config.csv_file_val,
                              file_path=val_file_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train_batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.val_batch_size)
    return train_loader, val_loader


def load_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载预训练
    if config.resume_weights:
        net = network()
        model_file = os.path.join("./models/", config.model_dir, 'weights/model-%d.pth' % config.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        config.begin_epoch = config.resume_weights
        net.load_state_dict(check_point)
        print(f'从epoch{config.resume_weights}开始训练')
    elif config.pretrained_path:
        net = network(pretrained_path=config.pretrained_path)
        print('使用预训练权重')

    # 将模型写入tensorboard
    init_img = torch.randn((1, 3, 224, 224))
    # init_img = torch.((1, 3, 224, 224))
    tb_writer.add_graph(net, init_img)

    # 优化器
    params = []
    if config.train_require_layers:
        for name, mdl in net.named_children():
            if name in config.train_require_layers:
                mdl.requires_grad_(True)
            else:
                mdl.requires_grad_(False)
    else:
        for name, mdl in net.named_children():
            mdl.requires_grad_(True)
    return net


def train_cele(net, train_loader, val_loader):
    '''
        args:
            net: 网络
            train_loader: 训练集
            val_loader: 验证集
        return:
            train_l: 训练loss组成的list
            val_l: 验证loss组成的list
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    msg = "start training on " + str(device)
    logging.info(msg)

    for epoch in range(config.begin_epoch, config.num_epochs):
        epoch += 1
        optimizer = make_optimizer(config, net)
        trainer = Trainer(config)
        train_loss, train_acc = trainer(net, optimizer, train_loader, epoch, config)
        evaluator = Evaluator(config)
        val_loss, val_acc = evaluator(net, val_loader, epoch, config)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), "./models/" + config.model_dir + "/weights/model-{}.pth".format(epoch))
        msg = "[train epoch {}] train loss: {:.3f}, train acc: {:.3f} val loss: {:.3f}, val acc: {:.3f}" \
              "".format(epoch, train_loss, train_acc, val_loss, val_acc)
        logging.info(msg)

    return train_loss, val_loss


if __name__ == '__main__':
    # 加载参数
    config, network = set_params()
    # tensorboard文件
    tb_writer = SummaryWriter(log_dir='runs/' + config.model_dir)
    # 加载数据集
    train_loader, val_loader = get_dataloader(config)
    # 加载预训练权重
    net = load_model(config)
    # 开始训练
    train_l, val_l = train_cele(net, train_loader, val_loader)
    logging.info('train_l:' + str(train_l) + 'val_l: ' + str(val_l))