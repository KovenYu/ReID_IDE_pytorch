import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import math


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_activation='relu'):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.last_activation = last_activation

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        # basicblock = F.leaky_relu(basicblock, 0.1, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.last_activation == 'tanh':
            return F.tanh(residual + basicblock)
        else:
            return F.relu(residual + basicblock, inplace=True)
        # return F.leaky_relu(residual + basicblock, 0.1, inplace=True)


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, block, depth, num_classes, last_activation='relu'):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
    """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes
        self.last_activation = last_activation

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2, self.last_activation)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d((36, 14))
        self.fc_final = nn.Linear(64 * block.expansion, num_classes, bias=False)
        self.bn_final = nn.BatchNorm1d(num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                # m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, last_activation='relu'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_activation == 'relu':
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
        else:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last_activation=last_activation))

        return nn.Sequential(*layers)

    def forward(self, x, is_mid_maps=False):
        if not is_mid_maps:
            x = self.conv_1_3x3(x)
            x = F.relu(self.bn_1(x), inplace=True)
            x = self.stage_1(x)
            x = self.stage_2(x)
        mid_maps = x
        maps = self.stage_3(mid_maps)
        x = self.avgpool(maps)
        feature = x.view(x.size(0), -1)
        predictions = self.fc_final(feature)
        predictions = self.bn_final(predictions)
        return maps, feature, predictions, mid_maps


def resnet20(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet32(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 56, num_classes, last_activation='tanh')
    return model


def resnet110(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model


class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x