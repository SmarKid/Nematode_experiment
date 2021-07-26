import sys
import unittest
from unittest.case import TestCase
import torchvision
import torch
sys.path.append('../')
from celegans_dataset import *

class TestCelegansDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = 'E:\workspace\线虫数据集\分类数据集'
        self.csv_tar_path = './csv_files'

    def tearDown(self) -> None:
        return super().tearDown()
    
    
    
    def test_generate_C_elegans_csv(self):
        generate_C_elegans_csv(self.root_dir, self.csv_tar_path, num_val=600, shuffle=True)

    def test_CelegansDataset(self):
        csv_file_train = '.\\test\csv_files\cele_df_train.csv'
        csv_file_val = '.\\test\csv_files\cele_df_val.csv'
        labels_name_required = 'shoot_days'
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(400, 600))
        ])
        train_set = CelegansDataset(labels_name_required, csv_file_val, self.root_dir, transform=trans)
        val_set = CelegansDataset(labels_name_required, csv_file_train, self.root_dir)
        batch_size = 5
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
        print('********')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCelegansDataset("test_CelegansDataset"))   
    unittest.TextTestRunner(verbosity=2).run(suite)