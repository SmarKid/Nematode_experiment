import torch.utils.data
from torchvision import models
import torchvision
import time
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from celegans_dataset import CelegansDataset

def evaluate_cele_accuracy(data_iter, net,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in data_iter:
            X = batch['image']
            y = batch['label']
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += net(X.to(device) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_cele(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch in train_iter:
            X = batch['image'].to(device)
            y = batch['label'].to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_cele_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

if __name__ == '__main__':
    # 1. 读取训练集和验证集
    csv_file_test = 'E:\workspace\线虫数据集\图片整理\cele_df_test.csv'
    csv_file_val = 'E:\workspace\线虫数据集\图片整理\cele_df_val.csv'
    root_dir = 'E:\workspace\线虫数据集\图片整理'
    labels_name_required = 'part'

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(400, 600))
    ])

    val_set = CelegansDataset(labels_name_required, csv_file_test, root_dir)
    train_set = CelegansDataset(labels_name_required, csv_file_val, root_dir, transform=trans)
    batch_size = 5
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # 测试数据读取
    # k = 1
    # for b in train_loader:
    #     print(b['image'].shape)
    #     print(b['label'].shape)
    #     k -= 1
    #     if k < 0:
    #         break
    # print('over')

    net = models.alexnet()
    net.classifier[6] = torch.nn.Linear(4096, 3)
    lr, num_epochs = 0.001, 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_cele(net, train_loader, val_loader, batch_size, optimizer, device, num_epochs)
