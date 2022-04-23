import numpy as np
import random
from skimage.segmentation import slic
import scipy.io as scio

import sklearn.cluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from hyperVca import hyperVca
from loadhsi import loadhsi

seed = 0
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_seed(seed)


case = ['ridge', 'synthetic']
case = case[0]
if case == 'ridge':
    data = scio.loadmat('dataset/JasperRidge2_R198.mat')
    Y = data['Y']
    Y = Y / np.max(Y)
    image = np.reshape(Y, [198, 100, 100])
    # image = image[:, 0:300, 0:300]
    image = np.swapaxes(image, 0, 2)
    image = image[:, :, [30, 20, 10]] * 3
    n_segments = 600
    compactness = 10.
    Channel = 198
    W = 100
    H = 100
    times = 250
    subNum = 200
    bundle_file = 'bundles_ridge.mat'
elif case == 'synthetic':
    data = scio.loadmat('dataset/usgs_new.mat')
    Y = data['hsi_20db']
    Y = Y / np.max(Y)
    image = np.reshape(Y, [175, 100, 100])
    image = np.swapaxes(image, 0, 2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image[:, :, [30, 20, 10]] * 1.5


    n_segments = 600
    compactness = 40.
    Channel = 175
    W = 100
    H = 100
    times = 250
    subNum = 200

    bundle_file = 'bundles_synthetic.mat'




segments = slic(image, n_segments=n_segments, start_label=1, max_iter=100, compactness=compactness)


Y, A_true, P = loadhsi(case)


Y = Y.reshape([Channel, W, H])
Y1 = []

for i in range(np.max(segments)):
    pos = np.where((i + 1) == segments)
    arg = Y[:, pos[0], pos[1]]
    arg = np.mean(arg, 1)
    Y1.append(arg)

Y1 = np.stack(Y1)


def create_cluster(Y, P=4):
    '''
    :param sparse_data: [N,L]  pixel num
    :param P:       number of endmembers
    :return:  M [P,L] endmember matrix
    '''
    # Manually override euclidean

    N, L = np.shape(Y)

    def cos_dist(X, Y=None, Y_norm_squared=None, squared=False):
        return cosine_similarity(X, Y)

    # sklearn.cluster.euclidean_distances = cos_dist
    scaler = StandardScaler(with_mean=False)
    sparse_data = scaler.fit_transform(Y)
    # sparse_data = Y
    kmeans = sklearn.cluster.KMeans(n_clusters=P)
    _ = kmeans.fit(sparse_data)
    M = np.zeros([P, L])
    for i in range(P):
        M[i, :] = np.mean(Y[kmeans.labels_ == i, :], axis=0)
    return M, kmeans.labels_


# region based VCA
EM_ = []
for i in range(times):
    index = np.random.randint(0, Y1.shape[0], subNum)
    Y_ = Y1[index, :]
    EM, _, _ = hyperVca(Y_.T, P)
    EM_.append(EM)

EM_ = np.stack(EM_, axis=1)
EM_ = EM_.reshape([Channel, -1])

M, y_pred = create_cluster(EM_.T, P)

EM_bundles = []
for i in range(P):
    temp = EM_[:, y_pred == i]
    EM_bundles.append(temp[:, 0:100].T)  # 100,L

EM_bundles = np.array(EM_bundles)
EM_bundles = np.swapaxes(EM_bundles, 0, 1)  # [100,P,L]


scio.savemat(bundle_file, {'bundleLibs': EM_bundles})


