from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderGCN(nn.Module):
    def __init__(self, P, Channel, z_dim, seq_l):
        super(EncoderGCN, self).__init__()
        #
        # encoder_z
        # gcn1-->gcn2-->fc1-->fc2(mu)
        #                 -->fc3(sigma)
        # encoder_a
        # gcn3-->gcn4-->gcn5
        self.P = P
        self.Channel = Channel

        # encoder_z
        self.gcn1 = nn.Linear(Channel, Channel)  # 32 * P
        self.lngcn1 = nn.LayerNorm(Channel)

        self.gcn2 = nn.Linear(Channel, Channel)  # 16 * P
        self.lngcn2 = nn.LayerNorm(Channel)

        self.fc1 = nn.Linear(Channel, P * 8)
        self.bnfc = nn.BatchNorm1d(P * 8)
        self.fc2 = nn.Linear(P * 8, z_dim)
        self.fc3 = nn.Linear(P * 8, z_dim)

        # encoder_a

        self.gcn3 = nn.Linear(Channel, Channel)
        self.lngcn3 = nn.LayerNorm(Channel)

        self.gcn4 = nn.Linear(Channel, Channel)
        self.lngcn4 = nn.LayerNorm(Channel)

        self.gcn5 = nn.Linear(Channel, P)


        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            # print(m)
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def encoder_z(self, x, L, mask):
        # x [B,N,Channel ]
        # L [B,N,N]
        # mask [B,N,1]
        x1 = self.gcn1(L @ x)

        x = self.lngcn1(x1+x)
        x = F.relu(x)
        mask_ = mask @ torch.ones(1, x.shape[-1], device=mask.device)
        x = x * mask_

        x1 = self.gcn2(L @ x)
        x = self.lngcn2(x1+x)
        x = F.relu(x)
        mask_ = mask @ torch.ones(1, x.shape[-1], device=mask.device)
        x = x * mask_


        x = torch.sum(x, dim=1)
        count = mask.sum(dim=1)
        x = x / count  # avg. pooling


        x = self.fc1(x)
        x = self.bnfc(x)
        x = F.relu(x)

        mu = self.fc2(x)
        log_var = self.fc3(x)

        return mu, log_var

    def encoder_a(self, x, L, mask):
        x1 = self.gcn3(L @ x)

        x = self.lngcn3(x1+x)
        x = F.relu(x)
        mask_ = mask @ torch.ones(1, x.shape[-1], device=mask.device)
        x = x * mask_

        x1 = self.gcn4(L @ x)
        x = self.lngcn4(x1+x)
        x = F.relu(x)
        mask_ = mask @ torch.ones(1, x.shape[-1], device=mask.device)
        x = x * mask_

        x = self.gcn5(L @ x)

        a = F.softmax(x, dim=2)  # [B,N,P] without mask
        return a

    def forward(self, x, L, mask):
        a = self.encoder_a(x, L, mask)
        mu, log_var = self.encoder_z(x, L, mask)
        return mu, log_var, a


