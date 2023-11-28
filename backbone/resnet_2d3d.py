## modified from https://github.com/kenshohara/3D-ResNets-PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

__all__ = [
    'ResNet2d3d', 
    'resnet18_2d3d', 'resnet34_2d3d', 'resnet50_2d3d', 'resnet101_2d3d', 'resnet152_2d3d', 'resnet200_2d3d', 
    'resnet18_3d', 'resnet34_3d', 'resnet50_3d', 'resnet101_3d', 'resnet152_3d', 'resnet200_3d'
    ]

def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(1,stride,stride),
        padding=1,
        bias=bias)

def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    # 1x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=bias)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3d, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2d, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet2d3d(nn.Module):
    def __init__(self, block, layers, first_channel=3):
        super(ResNet2d3d, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(first_channel, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=1)
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
            #    customized_stride = stride
                customized_stride = (1, stride, stride)

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False), 
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        
        return x

class ResNet3d(nn.Module):
    def __init__(self, block, layers, first_channel=3):
        super(ResNet3d, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(first_channel, 64, kernel_size=(1,7,7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=2)
        #self.lastconv = nn.Conv3d(2048, 256, kernel_size=(1,1,1), stride=(1,1,1), padding=0)
        #self.lastavg = nn.AvgPool3d(kernel_size=(7,7,7), stride=(1,1,1))
        self.lastavg = nn.AdaptiveAvgPool3d(output_size=(1,1,1))
        self.lastmlp = nn.Linear(2048, 256)
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # customized_stride to deal with 2d or 3d residual blocks
            if (block == Bottleneck2d) or (block == BasicBlock2d):
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride

            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=customized_stride, bias=False), 
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        #x = self.lastconv(x)
        x = self.lastavg(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = self.lastmlp(x)
        return x

# 2d-3d resnet
def resnet18_2d3d(**kwargs):
    model = ResNet2d3d([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [2, 2, 2, 2], **kwargs)
    return model

def resnet34_2d3d(**kwargs):
    model = ResNet2d3d([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet50_2d3d(**kwargs):
    model = ResNet2d3d([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet101_2d3d(**kwargs):
    model = ResNet2d3d([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 23, 3], **kwargs)
    return model

def resnet152_2d3d(**kwargs):
    model = ResNet2d3d([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 8, 36, 3], **kwargs)
    return model

def resnet200_2d3d(**kwargs):
    model = ResNet2d3d([Bottleneck2d, Bottleneck2d, Bottleneck3d, Bottleneck3d], 
                   [3, 24, 36, 3], **kwargs)
    return model


# 3d resnet
def resnet18_3d(**kwargs):
    model = ResNet2d3d([BasicBlock3d, BasicBlock3d, BasicBlock3d, BasicBlock3d], 
                   [2, 2, 2, 2], **kwargs)
    return model

def resnet34_3d(**kwargs):
    model = ResNet2d3d([BasicBlock3d, BasicBlock3d, BasicBlock3d, BasicBlock3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet50_3d(**kwargs):
    '''Constructs a ResNet-50 model. '''
    model = ResNet3d([Bottleneck3d, Bottleneck3d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 6, 3], **kwargs)
    return model

def resnet101_3d(**kwargs):
    model = ResNet2d3d([Bottleneck3d, Bottleneck3d, Bottleneck3d, Bottleneck3d], 
                   [3, 4, 23, 3], **kwargs)
    return model

def resnet152_3d(**kwargs):
    model = ResNet2d3d([Bottleneck3d, Bottleneck3d, Bottleneck3d, Bottleneck3d], 
                   [3, 8, 36, 3], **kwargs)
    return model

def resnet200_3d(**kwargs):
    model = ResNet2d3d([Bottleneck3d, Bottleneck3d, Bottleneck3d, Bottleneck3d], 
                   [3, 24, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    r18 = resnet18_2d3d()
    r50 = resnet50_3d()
    x = torch.randn(1, 3, 5, 224, 224)
    y = r18(x)
    r3d18 = resnet18_3d()
