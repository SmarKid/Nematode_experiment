import unittest
import numpy as np
import torch
import sys
import os
from torch._C import dtype
from sklearn import metrics
import sklearn.utils
class TestTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_array(self):
        print(np.abs(1-2))

    def test_sklearn_eval(self):
        import pandas as pd
        df = pd.read_csv('E:\\workspace\\python\\实验报告\\8.6\\inference on test\\8.6 epoch_8_output.csv')
        data_arr = df.to_numpy()
        y_pred = data_arr[:, 2].tolist()
        y_true = data_arr[:, 3].tolist()
        y_score = data_arr[:, 4].tolist()
        # top3_accuracy = metrics.top_k_accuracy_score(y_true=y_true, y_score=y_score, k=3, labels=None)
        report = metrics.classification_report(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
        pass

    def test_top_1_acc(self):
        from sklearn.utils import column_or_1d
        from sklearn.utils._encode import _unique
        
        labels = np.array([i for i in range(30)])
        labels = column_or_1d(labels)
        classes = _unique(labels)
        n_labels = len(labels)
        n_classes = len(classes)
        if not np.array_equal(classes, labels):
            raise ValueError("Parameter 'labels' must be ordered.")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTest("test_top_1_acc"))
    unittest.TextTestRunner(verbosity=2).run(suite)
