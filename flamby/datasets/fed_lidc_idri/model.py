import torch
import torch.nn as nn

"""
Based on https://github.com/black0017/MedicalZooPytorch
"""


class Baseline(nn.Module):
    """
    Implementation based on the VNet paper: https://arxiv.org/abs/1606.04797
    """

    def __init__(self, in_channels=1, dropout=True):
        super(Baseline, self).__init__()
        self.in_channels = in_channels

        self.in_tr = InputTransition(in_channels, 16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3, dropout=dropout)
        self.down_tr256 = DownTransition(128, 2, dropout=dropout)
        self.up_tr256 = UpTransition(256, 256, 2, dropout=dropout)
        self.up_tr128 = UpTransition(256, 128, 2, dropout=dropout)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


def passthrough(x, **kwargs):
    return x


class LUConv(nn.Module):
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

        self.bn1 = torch.nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, num_features=16):
        super(InputTransition, self).__init__()
        self.num_features = num_features
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(
            self.in_channels, self.num_features, kernel_size=5, padding=2
        )

        self.bn1 = torch.nn.BatchNorm3d(self.num_features)
        self.relu1 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        repeat_rate = int(self.num_features / self.in_channels)
        out = self.bn1(out)
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(outChans)

        self.do1 = passthrough
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.25)
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2
        )

        self.bn1 = torch.nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d(0.25)
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d(p=0.25)
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 1, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm3d(1)

        self.conv2 = nn.Conv3d(1, 1, kernel_size=1)
        self.relu1 = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # convolve 32 down to 1 channel
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.sigmoid(self.conv2(out))
        return out
