import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

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
    n_segments = 170
    compactness = 10.
    seg_file = 'segments_ridge.mat'
elif case == 'synthetic':
    data = scio.loadmat('dataset/usgs_new.mat')
    Y = data['hsi_20db']
    Y = Y / np.max(Y)
    image = np.reshape(Y, [175, 100, 100])
    image = np.swapaxes(image, 0, 2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image[:, :, [30, 20, 10]] * 1.5

    n_segments = 170
    compactness = 40.
    seg_file = 'segments_synthetic.mat'




segments = slic(image, n_segments=n_segments, start_label=1, max_iter=100, compactness=compactness)
# S/m=compatness

scio.savemat(seg_file, {'segments': segments})

fig = plt.figure("Superpixels -- %d segments" % (400))
plt.subplot(131)
plt.title('image')
plt.imshow(image)
plt.subplot(132)
plt.title('segments')
plt.imshow(segments)
plt.subplot(133)
plt.title('image and segments')
plt.imshow(mark_boundaries(image, segments, color=(0, 0, 0)))
plt.show()
