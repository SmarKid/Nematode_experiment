import sys
import unittest
from unittest.case import TestCase
import torchvision
import torch
sys.path.append('../')
from lib.celegans_dataset import *

class TestCelegansDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = 'F:/线虫分类数据集'
        self.csv_tar_path = './test_csv_files'
        if not os.path.exists(self.csv_tar_path):
            os.mkdir(self.csv_tar_path)

    def tearDown(self) -> None:
        return super().tearDown()
    
    
    
    def test_generate_C_elegans_csv(self):
        generate_C_elegans_csv(self.root_dir, self.csv_tar_path, num_val=10, num_train=10, shuffle=True)

    def test_CelegansDataset(self):
        csv_file_train = '.\csv_files\cele_df_train.csv'
        csv_file_val = '.\csv_files\cele_df_val.csv'
        labels_name_required = 'shoot_days'
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(400, 600))
        ])
        train_set = CelegansDataset(labels_name_required, csv_file_val, self.root_dir, transform=trans)
        val_set = CelegansDataset(labels_name_required, csv_file_train, self.root_dir)
        batch_size = 5
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        for batch in train_loader:
            print(batch['label'])
        print('********')

    def test_save_weights(self):
        net = torchvision.models.AlexNet()
        epoch = 0
        if (epoch + 1) % 1 == 0:
            filename = 'model_epoch_%d' % (epoch + 1)
            if not os.path.exists('./weights'):
                os.mkdir('./weights')
            weight_path = os.path.join('./weights', filename)
            torch.save(net.state_dict(), weight_path)
    
    def test_sample_distrabute(self):
        DATAPATH = 'E:\workspace\线虫数据集'
        skip_missing = True
        l = get_images_relative_path_list(DATAPATH, skip_missing_shootday=False)
        max = -1
        labels_name_required = 'shoot_days'
        cnt = {}
        for path in l:
            shoot_day = get_C_elegants_label(path, labels_name_required)
            if shoot_day not in cnt:
                cnt[shoot_day] = 1
            else:
                cnt[shoot_day] += 1
        after = dict(sorted(cnt.items(), key=lambda e: e[0]))
        for k, v in after.items():
            print(k, v)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  
        ax.plot(after.keys(), after.values(), label='num of sample')  
        ax.set_xlabel('key')  
        ax.set_ylabel('num')  
        ax.set_title("num of sample")  
        ax.legend() 
        plt.savefig('sample distribution.jpg')

    def test_distribution_of_csv(self):
        filepath = 'E:\workspace\python\\线虫实验\\test\csv_files\cele_df_val.csv' # csv path
        import pandas as pd
        df = pd.read_csv(filepath)
        df_path = df['path']
        l = list(df_path)

        cnt = {}
        labels_name_required = 'shoot_days'
        for path in l:
            shoot_day = get_C_elegants_label(path, labels_name_required)
            if shoot_day not in cnt:
                cnt[shoot_day] = 1
            else:
                cnt[shoot_day] += 1
        after = dict(sorted(cnt.items(), key=lambda e: e[0]))
        for k, v in after.items():
            print(k, v)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  
        ax.plot(after.keys(), after.values(), label='num of sample')  
        ax.set_xlabel('key')  
        ax.set_ylabel('num')  
        ax.set_title("num of sample")  
        ax.legend() 
        plt.savefig('cele_df_val distribution.jpg')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCelegansDataset("test_distribution_of_csv")) 
    unittest.TextTestRunner(verbosity=2).run(suite)