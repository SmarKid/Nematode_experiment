import unittest
import torch
import torchvision
from tools.train import evaluate_cele_accuracy
from lib.celegans_dataset import CelegansDataset
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_evaluate_cele_accuracy(self):
        
        csv_file_val = '.\\test_csv_files\cele_df_val.csv'
        labels_name_required = 'shoot_days'
        root_dir = 'F:/线虫分类数据集'
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(400, 600))
        ])
        val_set = CelegansDataset(labels_name_required, csv_file_val, root_dir, transform=trans)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=8)
        self.data_loader = val_loader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = torchvision.models.alexnet()
        net.classifier[6] = torch.nn.Linear(4096, 30)
        net = net.to(device)
        test_acc, test_loss = evaluate_cele_accuracy(data_loader=self.data_loader, net=net, device=device)
        print(test_acc, test_loss)
    
    def test_list(self):
        l = [1, 2, 3, 4]
        sl = str(l)
        r = range(2, 5)
        print(r)

    
    def test_plot(self):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        train_l = [1.5450524033569708, 1.5450524033569708, 1.4057851293222692, 2.113667633475327, 2.113667633475327, 0.918146289945618, 0.4738353956036451, 0.3669156188645014, 0.325840152053006, 0.31152566962125827, 0.27603441516470223, 0.33277355146601917, 0.22529075584089556, 0.2091030802460913, 0.2114223980322117, 0.19931965248613823, 0.1722205829415123, 0.1626451795663291, 0.21558568382752874, 0.39749173260316617]
        test_l = [0.4163840460777283, 0.3711990936597188, 0.37764606952667235, 0.3799197987715403, 0.3703549798329671, 0.3678174102306366, 0.36692224502563475, 0.36651184916496277, 0.3662908939520518, 0.3661498765150706, 0.3660673475265503, 0.3660121480623881, 0.36597381075223284, 0.36595014731089276, 0.36593371351559956, 0.36592671513557434, 0.365915744304657, 0.36591107805569967, 0.3659002908070882, 0.3659139116605123]
        test_l = [i * 8 for i in test_l]
        ax.plot(range(1, len(train_l) + 1), train_l, label='train_l')  # Plot some data on the axes.
        ax.plot(range(1, len(test_l) + 1), test_l, label='test_l')  # Plot some data on the axes.
        ax.set_xlabel('epoch')  # Add an x-label to the axes.
        ax.set_ylabel('loss')  # Add a y-label to the axes.
        ax.set_title("training plot")  # Add a title to the axes.
        ax.legend() 
        import time
        plt.savefig('./fig%s.jpg' % time.strftime("%Y-%m-%d", time.localtime()))

    def test_argparse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
        parser.add_argument('--resume_weights', '-r', default=None, type=int)
        args = parser.parse_args()
        model_root_dir = os.path.join('./models/', args.model_dir)
        sys.path.insert(0, model_root_dir)
        # 读取数据集
        from config import config
        from network import Network
        net = Network()
        print(net)
    
    def test_trans_old_dic_to_new(self):
        check_point = torch.load('.\models\\alexnet\weights\epoch_0.pth', map_location=device)
        weight = {'model': 'alexnet', 'epoch': 0, 'state_dict': check_point}
        path = './models/alexnet/weights/epoch_0.pth'
        torch.save(weight, path)
        print()



if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTrain("test_trans_old_dic_to_new"))  
    unittest.TextTestRunner(verbosity=2).run(suite)
