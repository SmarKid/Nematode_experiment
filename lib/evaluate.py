import numpy as np
import math

# def evaluate(data, index, firstn, num_class=30):
#     '''
#         args:
#             data:包含预测值与真实值的ndarray数组
#             index:二元数组，表示预测值与真实值所在位置
#             firstn:表示误差容许范围，即小于等于
#     '''
#     # pidx 和 tidx 分别表示预测值与真实值的index
#     pidx, tidx = index
#     # cnt_array 记录每个类别的TP, FP 和 FN
#     cnt_array = np.zeros(shape=(num_class, 3))
#     for i in range(data):
#         np.abs()
#         data[i, pidx] - data[i, tidx]
#         for j in range(num_class):
#             if np.abs(data[i, pidx] - data[i, tidx]) <= firstn:
#                 p
def accuracy(predict_label, true_label, tolerance=0):
    """
    args:
        predict_label: batch_size
        true_label: batch_size
    return:
        accuracy: float
    """
    if len(predict_label) != len(true_label):
        print('input error!')
        return -1
    delta = np.abs(predict_label - true_label)
    delta = np.where(delta <= tolerance, 1, 0)
    acc = delta.sum() / len(predict_label)
    return acc

def MAE(predict_label, true_label):
    if len(predict_label) != len(true_label):
        print('input error!')
        return -1
    delta = np.abs(predict_label - true_label)
    mae = delta.sum() / len(predict_label)
    return mae



            

