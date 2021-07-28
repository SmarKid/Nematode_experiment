import torch
import torchvision
import time
import sys
sys.path.append('f:\\zzk files\\workspace\\线虫实验')
print(sys.path)
from celegans_dataset import generate_C_elegans_csv
from celegans_dataset import CelegansDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

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
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_cele_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def evaluate_cele_accuracy(data_iter, net,
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    test_count = 0
    with torch.no_grad():
        for batch in data_iter:
            X = batch['image']
            y = batch['label']
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            test_count +=1
    return acc_sum / n

if __name__ == '__main__':
    # 产生数据集csv 训练集10个样本用来测试，测试集181个不用
    dataset_path = 'F:/图片整理'
    csv_tar_path = '.\\test\\test_dataset'
    generate_C_elegans_csv(dataset_path, csv_tar_path, num_val=10, shuffle=True, skip_missing_shootday=True)

    # 读取数据集
    csv_file_train = '.\\test\\test_dataset\cele_df_train.csv'
    csv_file_val = '.\\test\\test_dataset\cele_df_val.csv'
    labels_name_required = 'shoot_days'

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(400, 600))
    ])

    train_set = CelegansDataset(labels_name_required, csv_file_train, dataset_path, transform=trans)
    val_set = CelegansDataset(labels_name_required, csv_file_val, dataset_path)
    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)

    # 测试train_cele()函数
    net = torchvision.models.alexnet()
    net.classifier[6] = torch.nn.Linear(4096, 22)
    PATH = './net.pth'
    torch.save(net.state_dict(), PATH)
    lr, num_epochs = 0.001, 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_cele(net, train_loader, val_loader, batch_size, optimizer, device, num_epochs)
    PATH = './net.pth'
    torch.save(net.state_dict(), PATH) 