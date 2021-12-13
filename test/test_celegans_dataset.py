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
        self.root_dir = 'E:\workspace\线虫数据集\分类数据集'
        self.csv_tar_path = './test_csv_files'
        if not os.path.exists(self.csv_tar_path):
            os.mkdir(self.csv_tar_path)
        pass

    def tearDown(self) -> None:
        return super().tearDown()
    
    
    
    def test_generate_C_elegans_csv(self):
        generate_C_elegans_csv(self.root_dir, self.csv_tar_path, num_val=10, num_train=10, shuffle=True)

    def test_CelegansDataset(self):
        csv_file_train = 'csv files/cele_df_train.csv'
        csv_file_val = 'csv files/cele_df_val.csv'
        labels_name_required = 'shoot_days'
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(0.5, 2)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            normalize
        ])
        train_set = CelegansDataset(labels_name_required, csv_file_val, self.root_dir, transform=trans, label_type='pdf')
        val_set = CelegansDataset(labels_name_required, csv_file_train, self.root_dir)
        batch_size = 64
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        import tqdm
        data_loader = tqdm.tqdm(train_loader)
        i = 1
        print('\n********')
        for step, batch in enumerate(data_loader):
            label = batch['label']
            image = batch['image']
            pdf_label = batch['pdf_label']
            print('batch', i)
            print('label.shape', label.shape)
            print('image.shape', image.shape)
            print('pdf_label.shape', pdf_label.shape)
            i += 1
            if i >= 3:
                break
        print('********')

    def test_fast_load(self):
        file_path = 'D:\\Dataset\\线虫分类数据集transformed\\C_dataset_transformed.pth'
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.5, 1), ratio=(0.5, 2)),
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            normalize
        ])
        train_set = CelegansDataset(label_type='pdf', fast_load=True, file_path=file_path)
        batch_size = 64
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        import tqdm
        data_loader = tqdm.tqdm(train_loader)
        i = 1
        print('\n********')
        for step, batch in enumerate(data_loader):
            label = batch['label']
            image = batch['image']
            pdf_label = batch['pdf_label']
            print('batch', i)
            print('label.shape', label.shape)
            print('image.shape', image.shape)
            print('pdf_label.shape', pdf_label.shape)
            i += 1
            if i >= 10:
                break
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
        DATAPATH = 'E:\workspace\线虫数据�?'
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
        plt.title("线虫训练集样本分�?")
        plt.savefig('cele_df_train distribution pie.jpg')
        print()

    def test_compute_class_weight(self):
        '''
            计算类的权重并保存，用于平衡数据集偏�?
        '''
        filepath = 'E:\workspace\python\\线虫实验\\csv files\cele_df_train.csv' # csv path
        import pandas as pd
        df = pd.read_csv(filepath)
        df_path = df['path']
        l = list(df_path)
        sum_num = len(l)

        cnt = [1 for i in range(30)]
        labels_name_required = 'shoot_days'
        for path in l:
            shoot_day = get_C_elegants_label(path, labels_name_required)
            cnt[shoot_day] += 1
        num_list = np.array(cnt)
        
        # bottom = num_list * (1 / (np.max(num_list) + 100))
        # weights = 1 - np.sqrt(bottom)

        weights = 1 / sum_num / num_list

        weights = torch.tensor(weights, dtype=torch.float32)
        def normalization(data):
            data = np.ma.masked_invalid(data)
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range
        norm_weights = normalization(weights)
        torch.save(weights, 'class_weights.pt')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  
        ax.plot(range(30), norm_weights, label='weight')  
        ax.set_xlabel('label')  
        ax.set_ylabel('weight')  
        ax.set_title("train_class_norm_weights_distribution")  
        ax.legend() 
        plt.savefig('train_class_norm_weights_distribution.jpg')
        pass

    def test_pic(self):
        fig, ax = plt.subplots()  
        y = [-3.5456, -0.5751,  0.0191,  0.0858, -0.1081,  0.3676,  0.3517,  0.5990,
          0.3631,  0.3145,  0.3673,  0.4723,  0.5808,  0.7390,  0.5919,  0.5214,
          0.4162,  0.5394,  0.0973,  0.1094, -0.3411, -0.1097, -0.6302, -2.4475,
         -3.5475, -3.5422, -1.5136, -2.5208, -1.5947, -1.8931]
        
        ax.plot(range(30), y)  
        plt.show()
        pass
        # ax.set_xlabel('key')  
        # ax.set_ylabel('num')  
        # ax.set_title("num of sample")  
        # ax.legend() 
        # plt.savefig('cele_df_val distribution.jpg')
    
    def make_a_test_dataset(self):
        import shutil
        from natsort import ns, natsorted
        root_dir = 'E:\workspace\线虫数据集\分类数据集'
        tar_dir = 'E:\workspace\线虫数据集\分类测试数据集'
        cnt = [0 for i in range(30)]
        paths = []
        extense_name = ['.jpg', '.JPG', '.jpeg']
        def search_dir(root_path):
            path_list = os.listdir(root_path)
            path_list = natsorted(path_list,alg=ns.PATH)
            for f in path_list:
                if os.path.isdir(os.path.join(root_path, f)):
                    search_dir(os.path.join(root_path, f))
                elif os.path.isfile(os.path.join(root_path, f)):
                    if os.path.splitext(f)[1] not in extense_name:
                        continue
                    else:
                        label = get_C_elegants_label(f, 'shoot_days')
                        if label == -1: continue
                        if cnt[label] >= 10: continue
                        file_path = os.path.join(root_path, f)
                        tar_path = os.path.join(tar_dir, f)
                        shutil.copyfile(file_path,tar_path) 
                        paths.append(f)
                        cnt[label] += 1 
        search_dir(root_dir)
        print(cnt)
        columns = ['path']
        paths = natsorted(paths,alg=ns.PATH)
        cele_df_test = DataFrame(paths, columns=columns)
        cele_df_test.to_csv(os.path.join(tar_dir, 'cele_df_test.csv'))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCelegansDataset("test_fast_load"))
    unittest.TextTestRunner(verbosity=2).run(suite)