import os

seed = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTHONHASHSEED'] = str(seed)
import torch
import torch.utils
import torch.utils.data
from scipy.spatial.distance import cdist
import numpy as np
from model.model_VAEGAN import Decoder, Discriminator
from model.model_self_attention import EncoderSelfAttention
import random
from matplotlib import pyplot as plt

import scipy.io as scio

from loadhsi import loadhsi
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not torch.cuda.is_available():
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


set_seed(seed)

tic = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cases = ['ridge', 'synthetic']
case = cases[0]
print(case)
Y, A_true, P = loadhsi(case)
Channel = Y.shape[0]
N = Y.shape[1]
Iter = 1000
batchsz = 100
lr = 1e-4
lambda_kl = 0.001
z_dim = 4

training = True
load_weight = False

model_weights = './model_torch/weight/'
model_path = './model_torch/DGMSSU/out/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_weights):
    os.makedirs(model_weights)
encoder_weights = model_weights + 'encoder.pt'
decoder_weights = model_weights + 'decoder.pt'
disc_weights = model_weights + 'disc.pt'

if case == 'ridge':
    nump = 10
    width = 100
    height = 100
    seg_file = 'segments_ridge.mat'
    bundle_file = 'bundles_ridge.mat'
if case == 'synthetic':
    w, h = 10, 10  # [w,h]=[nump,nump]*patchSize
    width = 100
    height = 100
    pixelNum = width * height / w / h
    bundle_file = 'bundles_synthetic.mat'
    seg_file = 'segments_synthetic.mat'

EM_bundles = scio.loadmat(bundle_file)
EMr = EM_bundles['bundleLibs']
EMr = torch.tensor(EMr.astype(np.float32)).to(device)

Y = np.reshape(Y, (Y.shape[0], width, height))  # data cube [L,W,H]
Y = np.swapaxes(Y, 0, 2)

##############################################
# prepare superpixel data
segments = scio.loadmat(seg_file)['segments']
segments = segments.T

nums = segments.max()  # nums of superpixel
seq_l = max(np.bincount(segments.reshape([1, -1])[0]))  # find nums of pixels in largest superpixel=sequence length
L = []
data = []
mask = []
for i in range(1, nums + 1):
    pos = np.where(i == segments)
    data_ = Y[pos[0], pos[1], :]  # [n,C]
    padding_l = seq_l - data_.shape[0]
    mask_ = np.ones([1, data_.shape[0]])
    mask_ = np.concatenate([mask_, np.zeros([1, padding_l])], axis=1).T
    data_ = np.concatenate([data_, np.zeros([padding_l, Channel])])  # [n,C]-->[seq_l,C]
    dist = cdist(data_, data_, metric='euclidean')
    W = np.exp(-dist ** 2) - np.eye(seq_l)
    D = np.diag(W.sum(1) ** (-0.5))
    L_temp = W @ D
    L_ = D @ L_temp + np.eye(seq_l)

    pos_x = pos[0]
    pos_y = pos[1]
    # print(pos_x.shape)
    pos_x = np.concatenate([pos_x.reshape([-1, 1]), np.ones([padding_l, 1])])
    pos_y = np.concatenate([pos_y.reshape([-1, 1]), np.ones([padding_l, 1])])
    data_ = np.concatenate([data_, pos_x, pos_y], axis=1)

    data.append(data_)

    mask.append(mask_)
data = np.stack(data)

mask = np.stack(mask)

print('check shape of training data:')
print('data shape', data.shape)
print('mask shape', mask.shape)

epochs = Iter // (data.shape[0] // batchsz)  # 1000 steps

print('Iters:', epochs)

train_data = torch.tensor(data, dtype=torch.float32).to(device)
train_mask = torch.tensor(mask, dtype=torch.bool).to(device)
train_db = torch.utils.data.TensorDataset(train_data, train_mask)
train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True, drop_last=True)

## define model and optimizer
################################################################
encoder = EncoderSelfAttention(P, Channel, z_dim).to(device)
decoder = Decoder(P, Channel, z_dim).to(device)
discriminator = Discriminator(P, Channel).to(device)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0, 0.9))
optim_de_list = []
optim_disc_list = []
for i in range(P):
    optim_de_list.append(torch.optim.Adam(decoder.decoders_list[i].parameters(), lr=lr, betas=(0, 0.9)))
    optim_disc_list.append(
        torch.optim.Adam(discriminator.discriminators_list[i].parameters(), lr=lr, betas=(0, 0.9)))
################################################################

print('start training!')
if training == True:
    if load_weight == True:
        checkpoint = torch.load(encoder_weights)
        encoder.load_state_dict(checkpoint['model'])
        optimizer_encoder.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        checkpoint = torch.load(decoder_weights)
        decoder.load_state_dict(checkpoint['model'])
        for i in range(P):
            optim_de_list[i].load_state_dict(checkpoint['optimizer'][i])

        checkpoint = torch.load(disc_weights)
        discriminator.load_state_dict(checkpoint['model'])
        for i in range(P):
            optim_disc_list[i].load_state_dict(checkpoint['optimizer'][i])
        print('load model weight!')
    else:
        start_epoch = 0
    losses = []
    for epoch in range(start_epoch, start_epoch + epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        for step, data in enumerate(train_db):

            y_ = data[0].to(device)

            mask = data[1].to(device)

            y = y_[:, :, 0: -2]
            mu, log_var, a = encoder(y, mask)
            std = (log_var * 0.5).exp()
            eps = torch.randn(mu.shape, device=std.device)
            z = mu + eps * std
            em_tensor = decoder(z)

            y_hat = a @ em_tensor

            count = mask.sum()

            mask = ~mask
            y_hat = y_hat.masked_fill(mask, 0)

            loss_rec = ((y_hat - y) ** 2).sum() / count
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / y.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(2).to(device))

            loss_vae = loss_rec + lambda_kl * kl_div
            # 1 train encoder
            encoder.zero_grad()
            loss_vae.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 15)
            optimizer_encoder.step()

            # 2 train decoder/generator
            z = z.detach()
            em_tensor = decoder(z)

            pred_f = []
            for _ in range(P):
                disc = discriminator.discriminators_list[_]
                emf = em_tensor[:, _, :]
                pred_f.append(disc(emf))
            Pred_f = torch.cat(pred_f)

            loss_f1 = -Pred_f.mean()
            loss_gen = loss_f1

            decoder.zero_grad()
            loss_gen.backward()
            for i in range(P):
                torch.nn.utils.clip_grad_norm_(decoder.decoders_list[i].parameters(), 15)
                optim_de_list[i].step()

            # 3 train discriminator 5 steps
            for i in range(5):
                EMf = em_tensor.detach()
                pred_r = []
                pred_f = []
                for _ in range(P):
                    disc = discriminator.discriminators_list[_]
                    emr = EMr[:, _, :]  # [B,C]
                    emf = EMf[:, _, :]
                    pred_r.append(disc(emr))  # [[b,1],[b,1],..]
                    pred_f.append(disc(emf))

                Pred_r = torch.cat(pred_r)
                Pred_f = torch.cat(pred_f)

                loss_r = -Pred_r.mean()
                loss_f2 = Pred_f.mean()

                loss_disc = loss_r + loss_f2

                discriminator.zero_grad()
                loss_disc.backward()
                for i in range(P):
                    torch.nn.utils.clip_grad_norm_(discriminator.discriminators_list[i].parameters(), 15)
                    optim_disc_list[i].step()

        losses.append(loss_vae.detach().cpu().numpy())
        if (epoch + 1) % (epochs // 10) == 0:

            print("Encoder: epoch = {}, loss_rec={:.4f}, loss_klv={:.1f}, ".format(epoch + 1, float(loss_rec),
                                                                                   float(kl_div)))
            print("Decoder: loss_f={:.4f}".format(float(loss_f1)))
            print(
                "Discriminator: loss_r={:.4f}, loss_f={:.4f},loss_gp={:.4f}".format(float(loss_r), float(loss_f2),
                                                                                    float(loss_gp)))
            print('*' * 50)
            #################
            # save state
            state_enc = {'model': encoder.state_dict(), 'optimizer': optimizer_encoder.state_dict(), 'epoch': epoch}
            state_dec_optim = []
            state_disc_optim = []
            for i in range(P):
                state_dec_optim.append(optim_de_list[i].state_dict())
                state_disc_optim.append(optim_disc_list[i].state_dict())
            state_dec = {'model': decoder.state_dict(), 'optimizer': state_dec_optim, 'epoch': epoch}
            state_disc = {'model': discriminator.state_dict(), 'optimizer': state_disc_optim, 'epoch': epoch}
            torch.save(state_enc, encoder_weights)
            torch.save(state_dec, decoder_weights)
            torch.save(state_disc, disc_weights)
            ##############
            encoder.eval()
            decoder.eval()
            discriminator.eval()

            with torch.no_grad():
                train_data_ = train_data[:, :, 0:-2]
                mu, log_var, a = encoder(train_data_, train_mask)  # [B,n,P]
                std = (log_var * 0.5).exp()
                eps = torch.randn(mu.shape, device=std.device)
                z = mu + eps * std

                em_tensor = decoder(z)
                y_hat = a @ em_tensor

                A_hat = torch.zeros([width, height, P]).to(device)
                Y_hat = torch.zeros([width, height, Channel]).to(device)
                for i in range(nums):
                    pos = np.where((i + 1) == segments)
                    length = pos[1].shape[0]

                    A_hat[pos[0], pos[1], :] = a[i, 0:length]
                    Y_hat[pos[0], pos[1], :] = y_hat[i, 0:length]

                for i in range(P):
                    A_hat[:, :, i] = A_hat[:, :, i].T
                a = A_hat.reshape([N, P])

            scio.savemat(model_path + 'em_vae.mat', {'EM': em_tensor.cpu().numpy(), 'A': a.cpu().numpy(),
                                                     'Y_hat': y_hat.cpu().numpy()})  #
            scio.savemat(model_path + 'loss.mat', {'loss': losses})
            print('save results!')
            toc = time.time()
            print(epoch, toc - tic)

else:
    checkpoint = torch.load(encoder_weights)
    encoder.load_state_dict(checkpoint['model'])

    checkpoint = torch.load(decoder_weights)
    decoder.load_state_dict(checkpoint['model'])

    checkpoint = torch.load(disc_weights)
    discriminator.load_state_dict(checkpoint['model'])

    print('load model weight!')
    encoder.eval()
    decoder.eval()
    discriminator.eval()

    with torch.no_grad():
        y = train_data[101:102, :, 0:-2]
        y = y.repeat(100, 1, 1)

        mu, log_var, a = encoder(y, train_mask[101:102].repeat(100, 1, 1))  # [B,n,P]
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=std.device)
        z = mu + eps * std

        em_gen = decoder(z)
        for i in range(4):
            plt.figure(i + 1)
            plt.plot((em_gen[:, i, :].cpu().t()))
        for i in range(4):
            plt.figure(i + 1)

            plt.show()
        em_gen = em_gen.cpu().numpy()
        scio.savemat('dgmsan.mat', {'em_gen': em_gen})

    # Visualization of Attention Maps
    with torch.no_grad():
        train_data_ = train_data[:, :, 0:-2]
        mu, log_var, a = encoder(train_data_, train_mask)  # [B,n,P]
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=std.device)
        z = mu + eps * std

        em_tensor = decoder(z)  # [B,P,C]
        y_hat = a @ em_tensor  # [B,n,C]

        A_hat = torch.zeros([width, height, P]).to(device)
        Y_hat = torch.zeros([width, height, Channel]).to(device)
        for i in range(nums):
            pos = np.where((i + 1) == segments)
            length = pos[1].shape[0]

            A_hat[pos[0], pos[1], :] = a[i, 0:length]
            Y_hat[pos[0], pos[1], :] = y_hat[i, 0:length]

        for i in range(P):
            A_hat[:, :, i] = A_hat[:, :, i].T

        plotSuperpixel = 0
        pos = np.where((plotSuperpixel + 1) == segments)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.imshow(encoder.sa3.att.detach().cpu()[plotSuperpixel, 0],
                   interpolation=None)
        plt.axis([0, pos[0].shape[0] - 1, 0, pos[0].shape[0] - 1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        cc = plt.colorbar()
        cc.ax.tick_params(labelsize=14)
        # plt.savefig('sa3.eps')
        plt.show()

        plt.imshow(encoder.sa4.att.detach().cpu()[plotSuperpixel, 0],
                   interpolation=None)
        plt.axis([0, pos[0].shape[0] - 1, 0, pos[0].shape[0] - 1])
        cc = plt.colorbar()
        cc.ax.tick_params(labelsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # plt.savefig('sa4.eps')
        plt.show()

        plt.imshow(encoder.sa5.att.detach().cpu()[plotSuperpixel, 0],
                   interpolation=None)
        plt.axis([0, pos[0].shape[0] - 1, 0, pos[0].shape[0] - 1])
        cc = plt.colorbar()
        cc.ax.tick_params(labelsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.savefig('sa5.eps')
        plt.show()
toc = time.time()
print('time elapsed', toc - tic)
