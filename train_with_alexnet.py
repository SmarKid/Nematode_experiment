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
from celegans_dataset import CelegansDataset

file_handler = logging.FileHandler(filename='logging.log', mode='a', encoding='utf-8',)
stream_handler = logging.StreamHandler(sys.stderr)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %p',
    handlers=[file_handler,stream_handler],
    level=logging.INFO
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def train_cele(net, train_loader, val_loader, optimizer, device, config):
    net = net.to(device)
    msg = "start training on " + str(device)
    logging.info(msg)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    train_l = []
    test_l = []
    for epoch in range(config.begin_epoch, config.num_epochs + 1):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
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
        test_acc, test_loss = evaluate_cele_accuracy(val_loader, net, device)
        msg = 'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' \
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start)
        logging.info(msg)
        train_l.append(train_l_sum / batch_count)
        test_l.append(test_loss)
        if (epoch + 1) % 10 == 0:
            filename = 'Alexnet_model_epoch_%d.pth' % (epoch + 1)
            if not os.path.exists('./weights'):
                os.mkdir('./weights')
            weight_path = os.path.join('./weights', filename)
            torch.save(net.state_dict(), weight_path) 
            msg = weight_path + ' saved'
            logging.info(msg)
    return train_l, test_l

def evaluate_cele_accuracy(data_loader, net, device):
    acc_sum, n, test_l_sum = 0.0, 0, 0.0
    batch_count = 0
    with torch.no_grad():
        print('test')
        for batch in tqdm.tqdm(data_loader):
            X = batch['image']
            y = batch['label']
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                
                y_hat = net(X.to(device))
                loss = torch.nn.CrossEntropyLoss()
                test_l_sum += loss(y_hat, y.to(device)).item()
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=True).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            batch_count += 1
    return acc_sum / n, test_l_sum / batch_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, type=int)
    args = parser.parse_args()
    from config import config
    config.model_dir = args.model_dir
    config.resume_weights = args.resume_weights
    model_root_dir = os.path.join('./models/', config.model_dir)
    sys.path.insert(0, model_root_dir)
    # 读取数据集
    from network import Network

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(400, 600))
    ])

    train_set = CelegansDataset(config.labels_name_required, config.csv_file_train, config.root_dir, transform=trans)
    val_set = CelegansDataset(config.labels_name_required, config.csv_file_val, config.root_dir, transform=trans)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.train_batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.test_batch_size)

    net = Network()
    if config.resume_weights:
        model_file = os.path.join(config.model_dir, 'weights/epoch_%d' % config.resume_weights)
        check_point = torch.load(model_file)
        net.load_state_dict(check_point)
        config.begin_epoch = config.resume_weights + 1

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    try:
        train_l, test_l = train_cele(net, train_loader, val_loader, optimizer, device, config)
        logging.info('train_l:' + str(train_l) +'test_l: ' + str(test_l))
        # 画图
        fig, ax = plt.subplots()  
        ax.plot(range(1, len(train_l) + 1), train_l, label='train loss')  
        ax.plot(range(1, len(test_l) + 1), test_l, label='test loss')  
        ax.set_xlabel('epoch')  
        ax.set_ylabel('loss')  
        ax.set_title("training plot")  
        ax.legend() 
        plt.savefig('./fig%s.jpg' % time.strftime("%Y-%m-%d", time.localtime()))
    except Exception as e:
        msg = str(e)
        logging.error(msg, exc_info=True)  
    