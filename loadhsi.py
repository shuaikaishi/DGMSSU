import numpy as np
import scipy.io as scio

def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature

    '''
    if case == 'ridge':
        file = './dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h)
        Y = np.reshape(Y, [198, 100, 100])

        for i, y in enumerate(Y):
            Y[i] = y.T
        Y = np.reshape(Y, [198, 10000])
        GT_file = './dataset/JasperRidge2_end4.mat'
        A_true = scio.loadmat(GT_file)['A']
        A_true = np.reshape(A_true, (4, 100, 100))
        for i, A in enumerate(A_true):
            A_true[i] = A.T
        if np.max(Y) > 1:
            Y = Y / np.max(Y)
    elif case == 'synthetic':
        file = './dataset/usgs_new.mat'
        data = scio.loadmat(file)
        Y = data['hsi_20db']
        A_true = data['abundance']
        A_true = A_true.T.reshape(8, 100, 100)

    P = A_true.shape[0]

    Y = Y.astype(np.float32)
    A_true = A_true.astype(np.float32)

    return Y, A_true, P



