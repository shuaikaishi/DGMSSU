from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, out_features, dropout=0):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h



    def forward(self, queries, keys, values, mask):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).masked_fill(mask, 0).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1,
                                                                                            3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).masked_fill(mask, 0).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3,
                                                                                         1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).masked_fill(mask, 0).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1,
                                                                                           3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att=torch.cat([att[:,:,:,i].view(b_s,self.h,nq,1).masked_fill(mask.view(b_s,1,nq,1),0) for i in range(nq)],dim=3)

        att = torch.cat(
            [att[:, :, i, :].view(b_s, self.h, 1, nq).masked_fill(mask.view(b_s, 1, 1, nq),  -np.inf) for i in
             range(nq)], dim=2)

        att = torch.softmax(att, -1)
        self.att = att
        att = self.dropout(att)


        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = out.masked_fill(mask, 0)  # (b_s, nq, d_model)
        return out


class EncoderSelfAttention(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(EncoderSelfAttention, self).__init__()


        self.P = P
        self.Channel = Channel

        # encoder_z
        self.sa1 = ScaledDotProductAttention(Channel, Channel, Channel, 1, Channel)  # d_model, d_k, d_v, h,o_feature

        self.lnsa1 = nn.LayerNorm(Channel)
        self.sa2 = ScaledDotProductAttention(Channel, Channel, Channel, 1, Channel)

        self.lnsa2 = nn.LayerNorm(Channel)

        self.fc1 = nn.Linear(Channel, P * 8)
        self.bnfc = nn.BatchNorm1d(P * 8)
        self.fc2 = nn.Linear(P * 8, z_dim)
        self.fc3 = nn.Linear(P * 8, z_dim)

        # encoder_a

        self.sa3 = ScaledDotProductAttention(Channel, Channel, Channel, 1, Channel)

        self.lnsa3 = nn.LayerNorm(Channel)

        self.sa4 = ScaledDotProductAttention(Channel, Channel, Channel, 1, Channel)
        self.lnsa4 = nn.LayerNorm(Channel)

        self.sa5 = ScaledDotProductAttention(Channel, Channel, Channel, 1, P)

        self.init_parameters()

        self.fc11 = nn.Linear(Channel, Channel)
        self.ln11 = nn.LayerNorm(Channel)
        self.fc22 = nn.Linear(Channel, Channel)
        self.ln22 = nn.LayerNorm(Channel)
        self.fc33 = nn.Linear(Channel, Channel)
        self.ln33 = nn.LayerNorm(Channel)
        self.fc44 = nn.Linear(Channel, Channel)
        self.ln44 = nn.LayerNorm(Channel)
        self.fc55 = nn.Linear(Channel, P)
        self.lnsa5 = nn.LayerNorm(Channel)

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():

            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:

                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                # print(m)
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def encoder_z(self, x, mask):
        # x [B,N,Channel ]
        # mask [B,N,1]
        x1 = self.sa1(x, x, x, mask)
        x = self.lnsa1(x+x1)

        x1 = self.fc11(x)
        x1 = F.relu(x1).masked_fill(mask, 0)
        x = self.ln11(x1 + x)

        x1 = self.sa2(x, x, x, mask)
        x = self.lnsa2(x + x1)

        x1 = self.fc22(x)
        x1 = F.relu(x1).masked_fill(mask, 0)
        x = self.ln22(x1 + x)  # [b,n,C]


        x = torch.sum(x, dim=1)

        mask = ~mask
        count = mask.sum(dim=1)

        x = x / count  # avg. pooling


        x = self.fc1(x)
        x = self.bnfc(x)
        x = F.relu(x)

        mu = self.fc2(x)
        log_var = self.fc3(x)

        return mu, log_var
    def encoder_z1(self, x, mask):
        # x [B,N,Channel ]
        # mask [B,N,1]
        x1 = self.sa1(x, x, x, mask)
        x = self.lnsa1(x+x1)


        x = F.relu(x).masked_fill(mask, 0)


        x1 = self.sa2(x, x, x, mask)
        x = self.lnsa2(x + x1)


        x = F.relu(x).masked_fill(mask, 0)

        x = torch.sum(x, dim=1)

        mask = ~mask
        count = mask.sum(dim=1)

        x = x / count  # avg. pooling


        x = self.fc1(x)
        x = self.bnfc(x)
        x = F.relu(x)

        mu = self.fc2(x)
        log_var = self.fc3(x)

        return mu, log_var
    def encoder_a1(self, x, mask):
        x1 = self.sa3(x, x, x, mask)
        x = self.lnsa3(x + x1)

        x = F.relu(x).masked_fill(mask, 0)

        x1 = self.sa4(x, x, x, mask)
        x = self.lnsa4(x + x1)

        x = F.relu(x).masked_fill(mask, 0)
        x1 = self.sa5(x, x, x, mask)
        x = self.lnsa5(x + x1)
        x = self.fc55(x)

        a = F.softmax(x, dim=2)  # [B,N,P] without mask
        return a
    def encoder_a(self, x, mask):
        x1 = self.sa3(x, x, x, mask)
        x = self.lnsa3(x + x1)

        x1 = self.fc33(x)
        x1 = F.relu(x1).masked_fill(mask, 0)
        x = self.ln33(x1 + x)

        x1 = self.sa4(x, x, x, mask)
        x = self.lnsa4(x + x1)

        x1 = self.fc44(x)
        x1 = F.relu(x1).masked_fill(mask, 0)
        x = self.ln44(x1 + x)

        x1 = self.sa5(x, x, x, mask)
        x = self.lnsa5(x + x1)

        x = self.fc55(x)

        a = F.softmax(x, dim=2)  # [B,N,P] without mask
        return a

    def forward(self, x, mask):
        mask = ~mask
        a = self.encoder_a(x, mask)
        mu, log_var = self.encoder_z(x, mask)

        return mu, log_var, a

