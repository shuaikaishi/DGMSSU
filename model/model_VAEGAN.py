from torch import nn, autograd
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.nn.utils.spectral_norm import spectral_norm


# torch.nn.utils.weight_norm()
class Encoder(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(Encoder, self).__init__()
        self.P = P
        self.Channel = Channel
        # encoder z  fc1 -->fc5
        self.fc1 = (nn.Linear(Channel, 32 * P))
        # self.dp1=nn.Dropout(0.00)
        self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = (nn.Linear(32 * P, 16 * P))
        self.bn2 = nn.BatchNorm1d(16 * P)

        self.fc3 = (nn.Linear(16 * P, 4 * P))
        self.bn3 = nn.BatchNorm1d(4 * P)

        self.fc4 = (nn.Linear(4 * P, z_dim))
        self.fc5 = (nn.Linear(4 * P, z_dim))

        # encoder a
        self.fc9 = (nn.Linear(Channel, 32 * P))
        self.bn9 = nn.BatchNorm1d(32 * P)

        self.fc10 = (nn.Linear(32 * P, 16 * P))
        self.bn10 = nn.BatchNorm1d(16 * P)
        #
        self.fc11 = (nn.Linear(16 * P, 4 * P))
        self.bn11 = nn.BatchNorm1d(4 * P)

        self.fc12 = (nn.Linear(4 * P, 4 * P))
        self.bn12 = nn.BatchNorm1d(4 * P)

        self.fc13 = (nn.Linear(4 * P, 1 * P))  # get abundance

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            # print(m)
            classname = m.__class__.__name__  # 2
            if classname.find('Conv') != -1:  # 3
                nn.init.normal_(m.weight.data, 0.0, 0.02)  # 4
            elif classname.find('BatchNorm') != -1:  # 5
                nn.init.normal_(m.weight.data, 1.0, 0.02)  # 6
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def encoder_z(self, x):
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.fc2(h1)
        h1 = self.bn2(h1)
        h11 = F.relu(h1)

        h1 = self.fc3(h11)
        h1 = self.bn3(h1)
        h1 = F.relu(h1)

        mu = self.fc4(h1)
        log_var = self.fc5(h1)
        return mu, log_var

    def encoder_a(self, x):
        h1 = self.fc9(x)
        h1 = self.bn9(h1)
        h1 = F.relu(h1)

        h1 = self.fc10(h1)
        h1 = self.bn10(h1)
        h1 = F.relu(h1)

        h1 = self.fc11(h1)
        h1 = self.bn11(h1)
        h1 = F.relu(h1)

        h1 = self.fc12(h1)
        h1 = self.bn12(h1)

        h1 = F.relu(h1)
        h1 = self.fc13(h1)

        a = F.softmax(h1, dim=1)
        return a

    def forward(self, y):
        a = self.encoder_a(y)
        mu, log_var = self.encoder_z(y)
        return mu, log_var, a


class Decoder_Block(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(Decoder_Block, self).__init__()
        self.fc6 = (nn.Linear(z_dim, P * 16))
        self.bn6 = nn.BatchNorm1d(P * 16)

        self.fc7 = (nn.Linear(P * 16, P * 64))
        self.bn7 = nn.BatchNorm1d(P * 64)

        self.fc8 = (nn.Linear(P * 64, Channel))

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            # print(m)
            classname = m.__class__.__name__  # 2
            if classname.find('Conv') != -1:  # 3
                nn.init.normal_(m.weight.data, 0.0, 0.02)  # 4
            elif classname.find('BatchNorm') != -1:  # 5
                nn.init.normal_(m.weight.data, 1.0, 0.02)  # 6
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, z):
        h1 = self.fc6(z)
        h1 = self.bn6(h1)
        h1 = F.relu(h1)

        h1 = self.fc7(h1)
        h1 = self.bn7(h1)
        h1 = F.relu(h1)

        h1 = self.fc8(h1)
        em = torch.sigmoid(h1)
        return em


class Discriminator_Block(nn.Module):
    def __init__(self, Channel, P):
        super(Discriminator_Block, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(Channel, 64 * P))

        # self.bn1 = nn.BatchNorm1d(32 * P)

        self.fc2 = spectral_norm(nn.Linear(64 * P, 16 * P))
        # self.bn2 = nn.BatchNorm1d(16 * P)

        self.fc3 = spectral_norm(nn.Linear(16 * P, 1))
        # self.bn3 = nn.BatchNorm1d(4 * P)
        # self.fc4 = nn.Linear(4* P, 1)
        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            # print(m)
            classname = m.__class__.__name__  # 2
            if classname.find('Conv') != -1:  # 3
                nn.init.normal_(m.weight.data, 0.0, 0.02)  # 4
            elif classname.find('BatchNorm') != -1:  # 5
                nn.init.normal_(m.weight.data, 1.0, 0.02)  # 6
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self, inputs):
        h1 = self.fc1(inputs)
        # h1 = self.bn1(h1)
        h1 = F.relu(h1)

        h1 = self.fc2(h1)
        # h1 = self.bn2(h1)
        h11 = F.relu(h1)

        h1 = self.fc3(h11)
        # h1 = self.bn3(h1)
        # h1 = F.leaky_relu(h1, 0.00)

        # h1 = self.fc4(h1)
        # h1=torch.sigmoid(h1)
        return h1


class Decoder(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(Decoder, self).__init__()
        self.P = P
        self.decoders_list = nn.ModuleList()

        for _ in range(P):
            decoder = Decoder_Block(P, Channel, z_dim).to(device)
            self.decoders_list.append(decoder)

    def forward(self, z):
        em_list = []
        for _ in range(self.P):
            em = self.decoders_list[_](z)
            em_list.append(em)
            # self.discriminators_list[_](em)
        # [[B,C],[B,C],[B,C],...]
        em_tensor = torch.stack(em_list, dim=2)
        em_tensor = em_tensor.transpose(1, 2)  # [B,P,C]
        return em_tensor


class Discriminator(nn.Module):
    def __init__(self, P, Channel):
        super(Discriminator, self).__init__()
        self.P = P
        self.discriminators_list = nn.ModuleList()
        for _ in range(P):
            self.discriminators_list.append(Discriminator_Block(Channel, P).to(device))

    def gradient_penalty(self, D, xr, xf):

        batchsz = xf.shape[0]
        # only constrait for Discriminator
        xf = xf.detach()
        xr = xr.detach()

        # [b, 1] => [b, C]
        alpha = torch.rand(batchsz, 1).to(device)
        alpha = alpha.expand_as(xr)

        interpolates = alpha * xr + ((1 - alpha) * xf)
        interpolates.requires_grad_()

        disc_interpolates = D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(disc_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gp

    def forward(self, EMr, EMf):
        pred_r = []
        pred_f = []
        gp = 0
        for _ in range(self.P):
            disc = self.discriminators_list[_]
            emr = EMr[:, _, :]  # [B,C]
            emf = EMf[:, _, :]
            pred_r.append(disc(emr))  # [[b,1],[b,1],..]
            pred_f.append(disc(emf))
            gp = gp + self.gradient_penalty(disc, emr, emf)
        Pred_r = torch.cat(pred_r)
        Pred_f = torch.cat(pred_f)
        # print('Pred_r shape:',Pred_f.shape)
        gp = gp / self.P
        return Pred_r, Pred_f, gp

    def loss(self, Pred_r, Pred_f, gp):
        loss_r = -Pred_r.mean()
        loss_f = Pred_f.mean()
        loss_gp = gp
        return loss_r, loss_f, loss_gp


class VAEGAN(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(VAEGAN, self).__init__()
        self.P = P
        self.Channel = Channel

        self.encoder = Encoder(P, Channel, z_dim)
        self.decoder = Decoder(P, Channel, z_dim)
        self.discriminator = Discriminator(P, Channel)

        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            print(m)
            classname = m.__class__.__name__  # 2
            if classname.find('Conv') != -1:  # 3
                nn.init.normal_(m.weight.data, 0.0, 0.02)  # 4
            elif classname.find('BatchNorm') != -1:  # 5
                nn.init.normal_(m.weight.data, 1.0, 0.02)  # 6
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def forward(self, inputs):

        mu, log_var, a = self.encoder(inputs)
        # a [B,P]

        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        # print(z.device)
        em_tensor = self.decoder(z)

        a_tensor = a.view([-1, 1, self.P])  # [B,1,P]
        y_hat = a_tensor @ em_tensor
        y_hat = torch.squeeze(y_hat, dim=1)
        # y_hat = self.reparameterize(y_hat, self.noise_log_var)
        # a=a/a.sum(1).view(-1,1)#[b,P] / [b,1]
        # em_tensor=em_tensor*a.sum(1).view(-1,1,1) # [b,P,L] *[b,1]
        return y_hat, mu, log_var, a, em_tensor

    def loss(self, y_hat, y, mu, log_var, EMf, EMr):
        lam_mse = 0.1
        lam_kl = 0.1
        lam_gp = 0.3

        loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

        kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
        kl_div = kl_div.sum() / y.shape[0]

        # loss_enc
        loss_enc = loss_rec + lam_kl * kl_div

        # loss_disc   EMf,EMr [B,P,C]
        Pred_r, Pred_f, gp = self.discriminator(EMr, EMf)
        loss_r, loss_f, loss_gp = self.discriminator.loss(Pred_r, Pred_f, gp)

        loss_disc = loss_r + loss_f + lam_gp * loss_gp

        # loss_decoder
        # loss_dec = lam_mse * loss_rec - (1 - lam_mse) * loss_f
        loss_dec = - loss_f
        return loss_enc, loss_dec, loss_disc


if __name__ == '__main__':
    P, Channel, z_dim = 5, 200, 4
    device = 'cpu'
    model = Encoder(P, Channel, z_dim)
    input = torch.randn(10, Channel)
    mu, log_var, a = model(input)
    # print(' shape of y_hat: ', y_hat.shape)
    print('shape of mu: ', mu.shape)
    print('shape of var: ', log_var.shape)
    print('shape of a: ', a.shape)

    std = (log_var * 0.5).exp()
    eps = torch.randn(mu.shape, device=device)
    z = mu + eps * std
    model = Decoder(P, Channel, z_dim)
    em_tensor = model(z)
    print('shape of em:', em_tensor.shape)
    model = Discriminator(P, Channel)

    for _ in range(P):
        disc = model.discriminators_list[_]
        # emr = EMr[:, _, :]  # [B,C]
        emf = torch.rand(10, Channel)
        pred = disc(emf)
        print('disc out shape:', pred.shape)

    # decoder_weights='decoder.pt'
    # nn.init.constant_(model.discriminators_list[0].fc1.weight, 1)
    # state_dec = {'model': model.state_dict()}
    # torch.save(state_dec, decoder_weights)

    # checkpoint = torch.load(decoder_weights)
    # model.load_state_dict(checkpoint['model'])
    # print(model.discriminators_list[0].fc1.weight)

    # decoder_weights = 'decoder.pt'
    #
    # nn.init.constant_(model.discriminators_list[0].fc1.weight, 1)
    # model_disc = []
    # for i in range(P):
    #     model_disc.append(model.discriminators_list[i].state_dict())
    # state_disc = {'model': model_disc}
    # torch.save(state_disc, decoder_weights)
    #
    # checkpoint = torch.load(decoder_weights)
    # for i in range(P):
    #
    #     model.discriminators_list[i].load_state_dict(checkpoint['model'][i])
    # print(model.discriminators_list[0].fc1.weight)

    decoder_weights = 'decoder.pt'
    # nn.init.constant_(model.discriminators_list[0].fc1.weight, 1)
    # torch.save(model, decoder_weights)

    model = torch.load(decoder_weights)
    for i in model.modules():
        print(i)

    # print(model.discriminators_list[0].fc1.weight)

    # model()

    # print('em tensor shape: ', em.shape)

    # model.loss(y_hat, input, mu, log_var, em, em)

    # print(next(model.decoder.decoders_list[0].parameters()))
    # print(model.fc1.weight)
