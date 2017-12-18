import torch
import math
from torch import nn
import torchvision


class SingleBranchDecoder(nn.Module):
    def __init__(self, in_size=2048, out_channels=1):
        super(SingleBranchDecoder, self).__init__()
        self.in_size = in_size

        self.deconv = nn.Sequential(
            # (b, 2048, 1, 1)  ->  (b, 512, 4, 4)
            nn.ConvTranspose2d(self.in_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # (b, 512, 4, 4)  ->  (b, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            # (b, 256, 8, 8)  ->  (b, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # (b, 128, 16, 16)  ->  (b, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # (b, 64, 32, 32)  ->  (b, 32, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # (b, 32, 64, 64)  ->  (b, 1, 128, 128)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.autograd.Variable):
        # x's shape must be [b, self.in_size]
        assert x.size(1) == self.in_size

        # (b, in_size)

        x = x.view(x.size(0), self.in_size, 1, 1)
        x = self.deconv(x)

        # (b, out_channels, 128, 128)

        return x


class Decoder3d_32(nn.Module):
    def __init__(self, in_size=2048, out_channels=1):
        super(Decoder3d_32, self).__init__()
        self.in_size = in_size

        self.deconv = nn.Sequential(
            # 1 -> 4
            nn.ConvTranspose3d(self.in_size, 512, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(inplace=True),
            # 4 -> 8
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            # 8 -> 16
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            # 16 -> 32
            # (b, 1, 32, 32, 32)
            nn.ConvTranspose3d(128, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm3d(64),
            # nn.LeakyReLU(inplace=True),
            # # (b, 64, 32, 32)  ->  (b, 32, 64, 64)
            # nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm3d(32),
            # nn.LeakyReLU(inplace=True),
            # # (b, 32, 64, 64)  ->  (b, 1, 128, 128)
            # nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.autograd.Variable):
        # x's shape must be [b, self.in_size]
        assert x.size(1) == self.in_size

        # (b, in_size)

        x = x.view(x.size(0), self.in_size, 1, 1, 1)
        # (b, 2048, 1, 1)

        x = self.deconv(x)

        # (b, out_channels, 128, 128)

        return x


class Decoder3d_48(nn.Module):
    def __init__(self, in_size=2048, out_channels=1):
        super(Decoder3d_48, self).__init__()
        self.in_size = in_size

        self.deconv = nn.Sequential(
            # 1 -> 3
            nn.ConvTranspose3d(self.in_size, 512, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(inplace=True),
            # 3 -> 6
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(inplace=True),
            # 6 -> 12
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
            # 12 -> 24
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            # 24 -> 48
            # (b, 1, 48, 48, 48)
            nn.ConvTranspose3d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm3d(32),
            # nn.LeakyReLU(inplace=True),
            # # (b, 32, 64, 64)  ->  (b, 1, 128, 128)
            # nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x: torch.autograd.Variable):
        # x's shape must be [b, self.in_size]
        assert x.size(1) == self.in_size

        # (b, in_size)

        x = x.view(x.size(0), self.in_size, 1, 1, 1)
        # (b, 2048, 1, 1)

        x = self.deconv(x)

        # (b, out_channels, 128, 128)

        return x
