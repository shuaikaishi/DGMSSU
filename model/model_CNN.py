from torch import nn
import torch
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, P, Channel, z_dim):
        super(EncoderCNN, self).__init__()
        # conv1-->conv2--> fc1-->fc2(\mu)
        #                     -->fc3(\sigma)
        # conv3-->conv4-->conv5-->a
        # input shape: [B,C,W,H]
        self.P = P
        self.Channel = Channel

        # encoder_z
        self.conv1 = nn.Conv2d(Channel, Channel, (3, 3))
        self.bn1 = nn.BatchNorm2d(Channel)

        self.conv2 = nn.Conv2d(Channel, Channel, (3, 3))
        self.bn2 = nn.BatchNorm2d(Channel)

        self.fc1 = nn.Linear(Channel, P * 8)
        self.bnfc = nn.BatchNorm1d(P * 8)
        self.fc2 = nn.Linear(P * 8, z_dim)
        self.fc3 = nn.Linear(P * 8, z_dim)

        # encoder_a
        self.conv3 = nn.Conv2d(Channel, Channel, (3, 3))
        self.bn3 = nn.BatchNorm2d(Channel)

        self.conv4 = nn.Conv2d(Channel, Channel, (3, 3))
        self.bn4 = nn.BatchNorm2d(Channel)

        self.conv5 = nn.Conv2d(Channel, P, (3, 3))


        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            # print(m)
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:

                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def encoder_a(self, x):
        x = nn.ReplicationPad2d(padding=(1, 1, 1, 1))(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = nn.ReplicationPad2d(padding=(1, 1, 1, 1))(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = nn.ReplicationPad2d(padding=(1, 1, 1, 1))(x)
        x = self.conv5(x)

        a = F.softmax(x, dim=1)  # [b,P,10,10]
        return a

    def encoder_z(self, x):
        x = nn.ReplicationPad2d(padding=(1, 1, 1, 1))(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = nn.ReplicationPad2d(padding=(1, 1, 1, 1))(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.nn.AdaptiveAvgPool2d((1, 1))(x)  # [B,*,1,1]
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)

        x = self.fc1(x)
        x = self.bnfc(x)
        x = F.relu(x)

        mu = self.fc2(x)
        log_var = self.fc3(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = (log_var * 0.5).exp()
        eps = torch.randn(std.shape, device=std.device)
        return mu + eps * std

    def forward(self, y):
        a = self.encoder_a(y)
        mu, log_var = self.encoder_z(y)
        return mu, log_var, a


if __name__ == '__main__':
    P, Channel, z_dim = 5, 200, 4
    model = EncoderCNN(P, Channel, z_dim)
    input = torch.randn(10, Channel, 10, 10)
    mu, log_var, a = model(input)
    # print('Shape of y_hat:', y_hat.shape)
    print('Shape of mu:', mu.shape)
    print('Shape of log_var:', log_var.shape)
    print('Shape of abundance maps:', a.shape)

