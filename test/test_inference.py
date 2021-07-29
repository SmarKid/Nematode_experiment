import unittest
import numpy
import torch
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

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestTrain("test_array"))  
    unittest.TextTestRunner(verbosity=2).run(suite)
