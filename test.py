import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from loadhsi import loadhsi

cases = ['ridge', 'synthetic']
case = cases[0]

print(case)

model_path = './model_torch/DGMSSU/out/'
Y, A_true, P = loadhsi(case)
if case == 'ridge':
    nCol = 100
    nRow = 100

elif case == 'synthetic':
    nCol = 100
    nRow = 100
    lis = [0, 1, 2, 3, 4, 5, 6, 7]

nband = Y.shape[0]
N = Y.shape[1]


##########################################################
# plot loss
loss = scio.loadmat(model_path + 'loss.mat')['loss']
plt.loglog(loss[0])
plt.savefig('loss.png')
plt.show()
##########################################################
EM_hat = scio.loadmat(model_path + 'em_vae.mat')['EM']
A_hat = scio.loadmat(model_path + 'em_vae.mat')['A']
Y_hat = scio.loadmat(model_path + 'em_vae.mat')['Y_hat']

Y1 = Y.T

print(EM_hat.shape, A_hat.shape)


print('A_true shape', A_true.shape)
A_hat = np.reshape(A_hat, (nRow, nCol, P))

B = np.zeros((P, nRow, nCol))
for i in range(P):
    B[i] = A_hat[:, :, i]
A_hat = B
A_true = A_true.reshape([P, -1])
A_hat = A_hat.reshape([P, -1])


A_true = A_true.reshape([P, nCol, nRow])
A_hat = A_hat.reshape([P, nCol, nRow])

# plot Abundance maps
fig = plt.figure()
dev=np.zeros([P,P])
for i in range(P):
    for j in range(P):
        dev[i,j]=np.mean((A_hat[i,:,:]-A_true[j,:,:])**2 )
pos=np.argmin(dev,axis=0)

A_hat=A_hat[pos,:,:]
EM_hat = EM_hat[:, pos, :]
print(pos)
# plot abundance maps
for i in range(1, P + 1):
    plt.subplot(2, P, i + P)
    aaa = plt.imshow(A_true[i - 1], cmap='jet', interpolation='none')
    plt.axis('off')

    aaa.set_clim(vmin=0, vmax=1)
    plt.subplot(2, P, i)
    aaa = plt.imshow(A_hat[i - 1], cmap='jet',
                     interpolation='none')  # 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r'
    aaa.set_clim(vmin=0, vmax=1)
    plt.axis('off')
plt.show()



plt.figure()
for i in range(P):
    plt.subplot(2, (P + 1) // 2, i + 1)
    plt.plot(EM_hat[0:EM_hat.shape[0]: EM_hat.shape[0] // 100, i, :].T, 'c', linewidth=0.5)
    plt.xlabel('$\it{Bands}$', fontdict={'fontsize': 16})
    plt.ylabel('$\it{Reflectance}$', fontdict={'fontsize': 16})
    plt.axis([0, len(EM_hat[0, i, :]), 0, 1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.axis('off')



plt.figure()
for i in range(P):
    plt.subplot(2, (P + 1) // 2, i + 1)
    plt.plot(np.mean(EM_hat[:, i, :], axis=0))
plt.show()


print(A_hat.shape, A_true.shape)
armse = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))
print('aRMSE: ', armse)


