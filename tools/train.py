import torch
import torchvision
import time
import logging
import argparse
import sys
import os
from torchvision import models
import tqdm
import matplotlib.pyplot as plt
sys.path.append('../')
from lib.celegans_dataset import CelegansDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# tensorboard文件
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter(log_dir='runs')

# 配置日志
file_handler = logging.FileHandler(filename='./log/logging.log', mode='a', encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p',
    handlers=[file_handler,stream_handler],
    level=logging.ERROR
)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    '''
        训练一个epoch
    '''
    model.train()
    class_weight = torch.load('class_weights.pt').to(device)
    loss_function = torch.nn.CrossEntropyLoss(class_weight)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm.tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data['image'], data['label']
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        logging.info(data_loader.desc)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def train_cele(net, train_loader, val_loader, optimizer, device, config):
    '''
        args:
            net: 网络
            train_loader: 训练集
            val_loader: 验证集
            optimizer: 优化器
            device: 设备
            config: 配置
        return:
            train_l: 训练loss组成的list
            val_l: 验证loss组成的list
    '''
    # batch_count, train_l_sum, train_acc_sum, n, start = 0, 0.0, 0.0, 0, time.time()
    net = net.to(device)
    msg = "start training on " + str(device)
    logging.info(msg)
    
    for epoch in range(config.begin_epoch, config.num_epochs):
        
        epoch += 1
        train_loss, train_acc = train_one_epoch(net, optimizer, train_loader, device, epoch)
        val_acc, val_loss = evaluate(val_loader, net, device)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        logging.info(msg)

def evaluate(model, data_loader, device, epoch):
    '''
        args:
        return:
            val_loss: 验证loss
            test_accuracy: 验证精度
    '''
    class_weight = torch.load('class_weights.pt').to(device)
    loss_function = torch.nn.CrossEntropyLoss(class_weight)

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data['image'], data['label']
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        logging.info(data_loader.desc)                                                               

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

if __name__ == '__main__':
    
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

    # 数据loader
    train_set = CelegansDataset(config.labels_name_required, config.csv_file_train, 
                config.root_dir, transform=config.trans['train_trans'])
    val_set = CelegansDataset(config.labels_name_required, config.csv_file_val, 
                config.root_dir, transform=config.trans['val_trans'])
    batch_size = config.train_batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_set, num_workers=nw, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, num_workers=nw, batch_size=batch_size)

    # 加载预训练
    if config.resume_weights:
        net = network()
        model_file = os.path.join("./models/", config.model_dir, 'weights/epoch_%d.pth' % config.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        config.begin_epoch = config.resume_weights
        net.load_state_dict(check_point['state_dict'])
    elif config.pretrained_path:
        net = network(pretrained_path=config.pretrained_path)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 400, 600), device=device)
    tb_writer.add_graph(net, init_img)

    # 优化器
    params = []
    for name, model in net.named_children():
        if name in config.train_require_layers:
            params.append({'params': model.parameters()})

    optimizer = torch.optim.Adam(params, lr=config.learning_rate, 
            weight_decay=config.weight_decay)

    # 开始训练
    try:
        train_l, val_l = train_cele(net, train_loader, val_loader, optimizer, device, config)
        logging.info('train_l:' + str(train_l) +'val_l: ' + str(val_l))

    except Exception as e:
        msg = str(e)
        logging.error(msg, exc_info=True)  
    