import numpy as np

if __name__ == '__main__':
    l = [1, 2, 3, 4, 5, 6]
    n = np.array(l)
    ind = [0, 2, 4]
    n_ = [n[0],n[3],n[5]]
    print(np.array(n_).shape)