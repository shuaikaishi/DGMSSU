import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


def pca(X, d):
    N = np.shape(X)[1]
    xMean = np.mean(X, axis=1, keepdims=True)
    XZeroMean = X - xMean
    [U, S, V] = np.linalg.svd((XZeroMean @ XZeroMean.T) / N)
    Ud = U[:, 0:d]
    return Ud


def hyperVca(M, q):
    '''
    M : [p,N]
    '''
    L, N = np.shape(M)

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, 0:q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp)))
    snrEstimate = SNR
    # print('SNR estimate [dB]: %.4f' % SNR[0, 0])
    # Determine which projection to use.
    SNRth = 15 + 10 * np.log(q)+3#3
    # print(SNR,SNRth)
    if SNR > SNRth:
        d = q
        # [Ud, Sd, Vd] = svds((M * M.')/N, d);
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, 0:d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        # print(Xd.shape, u.shape, N, d)
        Y = Xd /  np.sum(Xd * u , axis=0, keepdims=True)

    else:
        d = q - 1
        r_bar = np.mean(M.T, axis=0, keepdims=True).T
        Ud = pca(M, d)

        R_zeroMean = M - r_bar
        Xd = Ud.T @ R_zeroMean
        # Preallocate memory for speed.
        # c = np.zeros([N, 1])
        # for j in range(N):
        #     c[j] = np.linalg.norm(Xd[:, j], ord=2)
        c = [np.linalg.norm(Xd[:, j], ord=2) for j in range(N)]
        # print(type(c))
        c = np.array(c)
        c = np.max(c, axis=0, keepdims=True) @ np.ones([1, N])
        Y = np.concatenate([Xd, c.reshape(1, -1)])
    e_u = np.zeros([q, 1])
    # print('*',e_u)
    e_u[q - 1, 0] = 1
    A = np.zeros([q, q])
    # idg - Doesntmatch.
    # print (A[:, 0].shape)
    A[:, 0] = e_u[0]
    I = np.eye(q)
    k = np.zeros([N, 1])

    indicies = np.zeros([q, 1])
    for i in range(q):  # i=1:q
        w = np.random.random([q, 1])

        # idg - Oppurtunity for speed up here.
        tmpNumerator = (I - A @ np.linalg.pinv(A)) @ w
        # f = ((I - A * pinv(A)) * w) / (norm(tmpNumerator));
        f = tmpNumerator / np.linalg.norm(tmpNumerator)

        v = f.T @ Y
        k = np.abs(v)

        k = np.argmax(k)
        A[:, i] = Y[:, k]
        indicies[i] = k

    indicies = indicies.astype('int')
    # print(indicies.T)
    if (SNR > SNRth):
        U = Ud @ Xd[:, indicies.T[0]]
    else:
        U = Ud @ Xd[:, indicies.T[0]] + r_bar

    return U, indicies, snrEstimate


if __name__ == '__main__':
    # data_file = './dataset/JasperRidge2_R198.mat'
    # GT_file = './dataset/JasperRidge2_end4.mat'
    # data_file='./dataset/Samson_R156.mat'
    # GT_file='./dataset/Samson_end3.mat'
    data_file = "./dataset/Urban_R162.mat"
    GT_file = './dataset/Urban_end4.mat'
    # data_file='./dataset/Paviauniversity_R103.mat'
    P = 4
    data = scio.loadmat(data_file)
    GT = scio.loadmat(GT_file)
    M = GT['M']
    A = GT['A']
    # print(np.mean(A,1))
    Y = data['Y']
    Y = Y / np.max(Y)

    U, indicies, snrEstimate = hyperVca(Y, P)
    # print(U.shape)
    plt.plot(U)

    plt.plot(M, 'k--')
    # plt.plot(Y[:,219*307+77])
    plt.show()


    # em=scio.loadmat('./model/vae/pre_train/EM_NCM_ridge.mat')['EM_NCM']
    # print(em.shape,M.shape)
    from ELMM.FCLSU import FCLSU
    a=FCLSU(Y,M)
    print(a.shape,A.shape)
    aRMSE=np.mean(np.sqrt(np.sum((A-a)**2,axis=0)))
    print(aRMSE)
    for i in range(P):
        plt.subplot(2,P,i+1)
        plt.imshow(a[i].reshape(307,307),cmap='jet')
        plt.subplot(2, P, i + 1+P)
        plt.imshow(A[i].reshape(307, 307), cmap='jet')
    plt.show()
