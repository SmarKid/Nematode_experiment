import unittest
import numpy
import torch
import sys
import os
from torch._C import dtype
class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_array(self):

        # output = torch.randn(8, 30)
        # print(output)
        # label = torch.tensor([1, 2, 3, 4, 5, 13, 24, 8], dtype=torch.int64)
        # label = label.resize(8, 1)
        # conf = torch.gather(output, 1, label)
        # print(conf)
        
        import numpy as np
        output = np.random.randn(8, 30)
        print(output)
        label = np.array([1, 2, 3, 4, 5, 13, 24, 8])
        y = output[range(len(output)),  label]
        print(y)



        # t = torch.tensor([[1, 2, 3], [3, 4, 5]])
        # out = torch.gather(t, 1, torch.tensor([[2], [0]]))
        # print(out)
    
    def test_inference(self):
        model_root_dir = os.path.join('./models/', 'efficientnetV2')
        sys.path.insert(0, model_root_dir)
        from network import network
        net = network()
        X = torch.rand((5, 3, 300, 300))
        out = net(X)
        pass

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTrain("test_inference"))  
    unittest.TextTestRunner(verbosity=2).run(suite)
