import torch.nn as nn
import torch.autograd as autograd
import torch
import math


class BottleneckGenerator(nn.Module):
    def __init__(self, is_transfer_net=True):
        super(BottleneckGenerator, self).__init__()

        self.is_transfer_net = is_transfer_net

        self.down_1 = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),  # 36, 14
                                    nn.BatchNorm2d(48),
                                    nn.LeakyReLU(0.2))
        self.down_2 = nn.Sequential(nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),  # 18, 7
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2))
        self.down_3 = nn.Sequential(nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),  # 9, 4
                                    nn.BatchNorm2d(96),
                                    nn.LeakyReLU(0.2))
        self.down_4 = nn.Sequential(nn.Conv2d(96, 128, 3, stride=2, padding=1, bias=False),  # 5, 2
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))
        self.up_1 = nn.Sequential(nn.ConvTranspose2d(128, 96, (3, 4), stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(192, 64, (4, 3), stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(128, 48, 4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(48),
                                  nn.ReLU())
        self.up_4 = nn.Sequential(nn.ConvTranspose2d(96, 32, 4, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.Tanh())
        self.mid_1 = nn.Sequential(nn.Conv2d(128, 256, (5, 2), bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2))
        self.mid_2 = nn.Sequential(nn.Conv2d(256, 64, (1, 1), bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2))
        self.mid_3 = nn.Sequential(nn.ConvTranspose2d(64, 128, (5, 2), bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, mid_maps, is_target=True):
        d1 = self.down_1(mid_maps)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        if self.is_transfer_net and is_target:
            m3 = d4
            features = None
        else:
            m1 = self.mid_1(d4)
            m2 = self.mid_2(m1)
            m3 = self.mid_3(m2)
            features = m2
        u1 = self.up_1(m3)
        u2 = self.up_2(torch.cat((u1, d3), dim=1))
        u3 = self.up_3(torch.cat((u2, d2), dim=1))
        new_maps = self.up_4(torch.cat((u3, d1), dim=1))
        return new_maps, features


class LongneckGenerator(nn.Module):
    def __init__(self, is_transfer_net=True):
        super(LongneckGenerator, self).__init__()

        self.is_transfer_net = is_transfer_net

        self.down_1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 36, 14
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU())
        self.down_2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18, 7
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU())
        self.mid = nn.Sequential(ResidualBlock(128),
                                 ResidualBlock(128),
                                 ResidualBlock(128),
                                 ResidualBlock(128),
                                 ResidualBlock(128),
                                 ResidualBlock(128))
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                                  nn.InstanceNorm2d(64),
                                  nn.ReLU())
        self.up_4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                                  nn.InstanceNorm2d(32),
                                  nn.Tanh())
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, mid_maps, is_target=True):
        d1 = self.down_1(mid_maps)
        d2 = self.down_2(d1)
        m = self.mid(d2)
        features = m
        u3 = self.up_3(m)
        new_maps = self.up_4(u3)
        return new_maps, features


class ResidualBlock(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, n_channel, kernel_size, stride, padding)
        self.in1 = nn.InstanceNorm2d(n_channel)
        self.in2 = nn.InstanceNorm2d(n_channel)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size, stride, padding)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.in1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = identity + x
        return self.activation(x)


class Discriminator(nn.Module):
    """
    A: Similar structure to BottleneckGeneratorA,
    downsample twice from (36, 14) to (9, 4), then down to (4, 2) to (1, 1)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(64, 96, 3, stride=2, padding=1, bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(96, 1, 3, stride=2, padding=1, bias=False),
                                  nn.AvgPool2d((5, 2)),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.main(x)
        probability_true = x.view(x.size(0), -1)
        return probability_true


class GlobalAvgPool(nn.Module):
    """
    if num_class specified, return a prediction vector;
    otherwise, return a 64-by-1 tensor which is the result of the global average pooling.
    """
    def __init__(self, height, width, num_class=None):
        super(GlobalAvgPool, self).__init__()
        self.avgpool = nn.AvgPool2d((height, width))
        self.num_class = num_class
        if num_class is not None:
            self.classifier = nn.Linear(64, num_class)
            self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.num_class is not None:
            x = self.classifier(x)
            x = self.bn(x)
        return x


def main():

    g = LongneckGenerator()
    d = Discriminator()
    criterion = nn.MSELoss()
    input = autograd.Variable(torch.randn(5, 32, 72, 28), requires_grad=True)
    target = input.detach()
    new_maps, _ = g(input, is_target=False)
    GAP = nn.AvgPool2d((36, 14))
    x = GAP(new_maps)
    x = x.view(x.size(0), -1)
    probability_true = d(new_maps)
    loss = criterion(new_maps, target)
    pass


if __name__ == '__main__':
    main()