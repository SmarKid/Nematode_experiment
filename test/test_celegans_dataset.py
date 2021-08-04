import sys
import unittest
import torchvision
import torch
sys.path.append('../')
from unittest.case import TestCase
from numpy import multiply, sqrt
from numpy.lib.npyio import save
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

    def test_distribution_of_csv_pie(self):
        filepath = 'E:\workspace\python\\线虫实验\\test\csv_files\cele_df_train.csv' # csv path
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
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.pie(after.values(),labels=after.keys(),autopct='%1.1f%%',startangle=150)
        plt.title("线虫训练集样本分布")
        plt.savefig('cele_df_train distribution pie.jpg')
        print()

    def test_compute_class_weight(self):
        filepath = 'E:\workspace\python\\线虫实验\\test\csv_files\cele_df_train.csv' # csv path
        import pandas as pd
        df = pd.read_csv(filepath)
        df_path = df['path']
        l = list(df_path)

        cnt = [0 for i in range(30)]
        labels_name_required = 'shoot_days'
        for path in l:
            shoot_day = get_C_elegants_label(path, labels_name_required)
            cnt[shoot_day] += 1
        num_list = np.array(cnt)
        
        bottom = num_list * (1 / (np.max(num_list) + 100))
        weights = 1 - np.sqrt(bottom)
        weights = torch.tensor(weights, dtype=torch.float32)
        torch.save(weights, 'class_weights.pt')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  
        ax.plot(range(30), weights, label='weight')  
        ax.set_xlabel('key')  
        ax.set_ylabel('weight')  
        ax.set_title("weight_set_distribution")  
        ax.legend() 
        plt.savefig('weight_set_distribution.jpg')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCelegansDataset("test_compute_class_weight")) 
    unittest.TextTestRunner(verbosity=2).run(suite)