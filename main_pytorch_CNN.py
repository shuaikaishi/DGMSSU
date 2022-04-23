import os
seed = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTHONHASHSEED'] = str(seed)
import torch
import torch.utils
import torch.utils.data
import numpy as np
import random
from model.model_CNN import EncoderCNN
from model.model_VAEGAN import Decoder, Discriminator
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
batchsz = 100  # N//100
lr = 1e-4
z_dim = 4
lambda_kl = 0.001
Iter = 1000

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
    w, h = 10, 10
    width = 100
    height = 100
    pixelNum = width * height / w / h
    bundle_file = 'bundles_ridge.mat'

if case == 'synthetic':
    w, h = 10, 10
    width = 100
    height = 100
    pixelNum = width * height / w / h

    bundle_file = 'bundles_synthetic.mat'

Y = np.reshape(Y, (Y.shape[0], width, height))

EM_bundles = scio.loadmat(bundle_file)

EMr = EM_bundles['bundleLibs']
EMr = torch.tensor(EMr.astype(np.float32)).to(device)


def cut_hsi(Y):
    data = []
    for i in range(width // w):
        for j in range(height // h):
            data.append(Y[:, i * w:i * w + w, j * h:j * h + h])
    data = np.stack(data, axis=0)  # [100,198,10,10]
    return data


Y = cut_hsi(Y)
train_Y = Y
epochs = Iter // (train_Y.shape[0] // batchsz)  # 1000 steps
print('Iters:', epochs)

train_db = torch.tensor(train_Y)

print('training data shape: ', train_db.shape)

train_db = torch.utils.data.TensorDataset(train_db)
train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True, drop_last=True)
############################################
## define model and optimizer
encoder = EncoderCNN(P, Channel, z_dim).to(device)
decoder = Decoder(P, Channel, z_dim).to(device)
discriminator = Discriminator(P, Channel).to(device)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
optim_de_list = []
optim_disc_list = []
for i in range(P):
    optim_de_list.append(torch.optim.Adam(decoder.decoders_list[i].parameters(), lr=lr, betas=(0.5, 0.9)))
    optim_disc_list.append(
        torch.optim.Adam(discriminator.discriminators_list[i].parameters(), lr=lr, betas=(0.5, 0.9)))

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
    print('start training!')
    tic = time.time()
    for epoch in range(start_epoch, start_epoch + epochs):
        encoder.train()
        decoder.train()
        discriminator.train()

        for step, data in enumerate(train_db):

            y = data[0].to(device)
            mu, log_var, a = encoder(y)
            std = (log_var * 0.5).exp()
            eps = torch.randn(mu.shape, device=std.device)
            z = mu + eps * std
            em_tensor = decoder(z)

            em_tensor = em_tensor.view(em_tensor.shape[0], 1, 1, P, Channel)  # [b,1,1,p,c]

            a_tensor = a.view(-1, 1, P, a.shape[2], a.shape[3])  # [B,1,P,w,h]
            a_tensor = a_tensor.transpose(1, 3)
            a_tensor = a_tensor.transpose(2, 4)

            y_hat = a_tensor @ em_tensor
            y_hat = y_hat.transpose(1, 3)
            y_hat = y_hat.transpose(2, 4)
            y_hat = torch.squeeze(y_hat, dim=1)

            loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0] / pixelNum
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / y.shape[0]

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
                    emr = EMr[:, _, :]
                    emf = EMf[:, _, :]
                    pred_r.append(disc(emr))
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

            print("Encoder: epoch = {}, loss_rec={:.4f}, loss_klv={:.4f}".format(epoch + 1, float(loss_rec),
                                                                                 float(kl_div)))
            print("Decoder: loss_f={:.4f}".format(float(loss_f1)))
            print(
                "Discriminator: loss_r={:.4f}, loss_f={:.4f}".format(float(loss_r), float(loss_f2)))
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
                y = torch.tensor(Y).to(device)
                mu, log_var, a = encoder(y)

                std = (log_var * 0.5).exp()
                eps = torch.randn(mu.shape, device=std.device)
                z = mu + eps * std

                em_tensor = decoder(z)

                em_tensor = em_tensor.view(em_tensor.shape[0], 1, 1, P, Channel)

                a_tensor = a.view(-1, 1, P, a.shape[2], a.shape[3])
                a_tensor = a_tensor.transpose(1, 3)
                a_tensor = a_tensor.transpose(2, 4)

                y_hat = a_tensor @ em_tensor
                y_hat = y_hat.transpose(1, 3)
                y_hat = y_hat.transpose(2, 4)  # [b,1,c,w,h]
                y_hat = torch.squeeze(y_hat, dim=1)

                em_tensor = torch.squeeze(em_tensor)  # [B,P,C]

                A_hat = torch.zeros([P, width, height])
                for i in range(width // w):
                    for j in range(height // h):
                        A_hat[:, i * w:i * w + w, j * h:j * h + h] = a[height // h * i + j]

                a = A_hat.reshape([P, N]).t()

            scio.savemat(model_path + 'em_vae.mat', {'EM': em_tensor.cpu().numpy(), 'A': a.cpu().numpy(),
                                                     'Y_hat': y_hat.cpu().numpy()})  #
            scio.savemat(model_path + 'loss.mat', {'loss': losses})
            print('save results!')
            toc = time.time()
            print('epoch = {}, time elapsed = {:.2f}s'.format(epoch, toc - tic))

else:
    pass

toc = time.time()
print('time elapsed:', toc - tic)

encoder.eval()
decoder.eval()
discriminator.eval()

with torch.no_grad():
    y = torch.tensor(Y).to(device)
    mu, log_var, a = encoder(y)

    std = (log_var * 0.5).exp()
    eps = torch.randn(mu.shape, device=std.device)
    z = mu + eps * std

    em_tensor = decoder(z)
    em_tensor = em_tensor.view(em_tensor.shape[0], 1, 1, P, Channel)

    a_tensor = a.view(-1, 1, P, a.shape[2], a.shape[3])
    a_tensor = a_tensor.transpose(1, 3)
    a_tensor = a_tensor.transpose(2, 4)

    y_hat = a_tensor @ em_tensor
    y_hat = y_hat.transpose(1, 3)
    y_hat = y_hat.transpose(2, 4)
    y_hat = torch.squeeze(y_hat, dim=1)

    em_tensor = torch.squeeze(em_tensor)

    A_hat = torch.zeros([P, width, height])
    for i in range(width // w):
        for j in range(height // h):
            A_hat[:, i * w:i * w + w, j * h:j * h + h] = a[height // h * i + j]



    A_hat = A_hat.reshape([P, N]).numpy()
    scio.savemat(model_path + 'em_vae.mat', {'EM': em_tensor.data.cpu().numpy(),
                                             'A': A_hat.T,
                                             'Y_hat': y_hat.cpu().numpy()})
