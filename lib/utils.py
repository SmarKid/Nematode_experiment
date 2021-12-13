import sys

import torch
import tqdm
from lib.make_loss import make_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, config):
        self.loss_function = config.loss_function

    def __call__(self, *args):
        if self.loss_function in ['dldl_loss', 'KLDivLoss']:
            return self.train_one_epoch_dldl(*args)
        else:
            return self.train_one_epoch(*args)

    def train_one_epoch_dldl(self, model, optimizer, data_loader, epoch, config):
        """
            训练一个epoch
        """
        model.train()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        with tqdm.tqdm(data_loader) as data_loader:
            for step, data in enumerate(data_loader):
                images, label, dl_label = data['image'].to(device), data['label'].to(device), data['pdf_label'].to(device)
                optimizer.zero_grad()
                ld_hat = model(images)
                rank = torch.Tensor([i for i in range(30)]).to(device)
                predict_labels = torch.sum(ld_hat * rank, dim=1)
                sample_num += len(data['label'])
                accu_num += torch.sum(torch.abs(predict_labels - label.to(device)) <= 1)
                loss_function = make_loss(config)
                loss = loss_function(predict_labels, label, ld_hat, dl_label)
                loss.backward()
                accu_loss += loss.detach()
                avg_loss, avg_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
                data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       avg_loss,
                                                                                       avg_acc)

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

                optimizer.step()

        return avg_loss, avg_acc


    def train_one_epoch(self, model, optimizer, data_loader, epoch, config):
        """
                    训练一个epoch
                """
        model.train()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        optimizer.zero_grad()

        sample_num = 0
        with tqdm.tqdm(data_loader) as data_loader:
            for step, data in enumerate(data_loader):
                images, label = data['image'].to(device), data['label'].to(device)
                y_hat = model(images.to(device))
                sample_num += len(data['label'])
                pred_classes = torch.max(y_hat, dim=1)[1]
                accu_num += torch.eq(pred_classes, label.to(device)).sum()
                from lib.make_loss import make_loss
                loss_function = make_loss(config)
                loss = loss_function(y_hat, label)
                loss.backward()
                accu_loss += loss.detach()
                avg_loss, avg_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
                data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       avg_loss,
                                                                                       avg_acc)

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

                optimizer.step()
                optimizer.zero_grad()

        return avg_loss, avg_acc

class Evaluator:
    def __init__(self, config):
        self.loss_function = config.loss_function

    def __call__(self, *args):
        if self.loss_function in ['dldl_loss', 'KLDivLoss']:
            return self.evaluate_one_epoch_dldl(*args)
        else:
            return self.evaluate_one_epoch(*args)

    def evaluate_one_epoch_dldl(self, model, data_loader, epoch, config):
        '''
            args:
            return:
                val_loss: 验证loss
                test_accuracy: 验证精度
        '''

        model.eval()

        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(device)  # 累计损失

        sample_num = 0
        with tqdm.tqdm(data_loader) as data_loader:
            for step, data in enumerate(data_loader):
                images, label, dl_label = data['image'].to(device), data['label'].to(device), data['pdf_label'].to(device)
                sample_num += len(data['label'])
                ld_hat = model(images.to(device))
                rank = torch.Tensor([i for i in range(30)]).to(device)
                predict_labels = torch.sum(ld_hat * rank, dim=1)
                accu_num += torch.sum(torch.abs(predict_labels - label.to(device)) <= 1)
                loss_function = make_loss(config)
                loss = loss_function(predict_labels, label, ld_hat, dl_label)
                accu_loss += loss.detach()
                avg_loss, avg_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num

                data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       avg_loss,
                                                                                       avg_acc)

        return avg_loss, avg_acc

    def evaluate_one_epoch(self, model, data_loader, epoch, config):
        '''
            args:
            return:
                val_loss: 验证loss
                test_accuracy: 验证精度
        '''

        model.eval()

        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        accu_loss = torch.zeros(1).to(device)  # 累计损失

        sample_num = 0
        with tqdm.tqdm(data_loader) as data_loader:
            for step, data in enumerate(data_loader):
                images, label = data['image'].to(device), data['label'].to(device)
                sample_num += len(data['label'])
                output = model(images)
                pred_classes = pred_classes = torch.max(output, dim=1)[1]
                accu_num += torch.eq(pred_classes, label.to(device)).sum()
                loss_function = make_loss(config)
                loss = loss_function(output, label)
                accu_loss += loss.detach()
                avg_loss, avg_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num

                data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       avg_loss,
                                                                                       avg_acc)


        return avg_loss, avg_acc