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
tb_writer = SummaryWriter(log_dir='runs/elegans_experience/')

# 配置日志
file_handler = logging.FileHandler(filename='./log/logging.log', mode='a', encoding='utf-8',)
stream_handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p',
    handlers=[file_handler,stream_handler],
    level=logging.INFO
)

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
    net = net.to(device)
    msg = "start training on " + str(device)
    logging.info(msg)
    class_weight = torch.load(config.class_weights_path)
    loss = torch.nn.CrossEntropyLoss(class_weight)
    train_l = []
    val_l = []
    for epoch in range(config.begin_epoch, config.num_epochs):
        batch_count, train_l_sum, train_acc_sum, n, start = 0, 0.0, 0.0, 0, time.time()
        print('train %d epoch' % (epoch + 1))
        for batch in tqdm.tqdm(train_loader):
            X = batch['image'].to(device)
            y = batch['label'].to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_acc, val_loss = evaluate_cele_accuracy(val_loader, net, device, class_weight)
        train_loss, train_acc = train_l_sum / batch_count, train_acc_sum / n

        # add loss, acc and lr into tensorboard
        tags = ['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]['lr'], epoch)

        msg = 'epoch %d, train loss %.4f, val loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec' \
              % (epoch + 1, train_loss, val_loss, train_acc, val_acc, time.time() - start)
        logging.info(msg)

        train_l.append(train_loss)
        val_l.append(val_loss)

        # 没10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            filename = 'epoch_%d.pth' % (epoch + 1)
            weight_dir = os.path.join('./models' , config.model_dir , 'weights')
            if not os.path.exists(weight_dir):
                os.mkdir(weight_dir)
            weight_path = os.path.join(weight_dir, filename)
            weight = {'model': config.model_dir, 'epoch': epoch + 1, 'state_dict': net.state_dict()}
            torch.save(weight, weight_path) 
            msg = weight_path + ' saved'
            logging.info(msg)

    return train_l, val_l

def evaluate_cele_accuracy(data_loader, net, device, class_weight=None):
    '''
        args:
        return:
            val_loss: 验证loss
            test_accuracy: 验证精度
    '''
    acc_sum, n, val_l_sum, batch_count = 0.0, 0, 0.0, 0
    with torch.no_grad():
        print('val')
        for batch in tqdm.tqdm(data_loader):
            X = batch['image'].to(device)
            y = batch['label'].to(device)
            net.eval() # 评估模式, 这会关闭dropout
            
            y_hat = net(X)
            loss = torch.nn.CrossEntropyLoss(class_weight)
            val_l_sum += loss(y_hat, y).item()
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            net.train() # 改回训练模式
            n += y.shape[0]
            batch_count += 1
    return acc_sum / n, val_l_sum / batch_count

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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train_batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.val_batch_size)

    # 加载预训练
    os.environ['TORCH_HOME'] = config.TORCH_HOME
    net = network(pretrained=config.pretrained)
    if config.resume_weights:
        model_file = os.path.join("./models/", config.model_dir, 'weights/epoch_%d.pth' % config.resume_weights)
        check_point = torch.load(model_file, map_location=device)
        config.begin_epoch = config.resume_weights
        net.load_state_dict(check_point['state_dict'])

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

        # 画图
        fig, ax = plt.subplots()  
        ax.plot(range(1, len(train_l) + 1), train_l, label='train loss')  
        ax.plot(range(1, len(val_l) + 1), val_l, label='val loss')  
        ax.set_xlabel('epoch')  
        ax.set_ylabel('loss')  
        ax.set_title("training plot")  
        ax.legend() 
        plt.savefig('./fig%s.jpg' % time.strftime("%Y-%m-%d", time.localtime()))
    except Exception as e:
        msg = str(e)
        logging.error(msg, exc_info=True)  
    