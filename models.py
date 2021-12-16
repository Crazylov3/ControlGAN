import torch
import torch.nn as nn


class Residual_conv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, size_layer_norm=None):
        super(Residual_conv, self).__init__()
        if batch_norm is False and size_layer_norm is None:
            raise Exception("Bad format of input")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = None
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
            self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels) if batch_norm else nn.InstanceNorm2d(out_channels, affine=True)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.InstanceNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.InstanceNorm2d(out_channels, affine=True)
        self.bn_ = nn.BatchNorm2d(out_channels) if batch_norm else nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, X):
        Y = nn.LeakyReLU(0.1)(self.bn1(self.conv1(X)))
        Y = nn.LeakyReLU(0.1)(self.bn2(self.conv2(Y)))
        if self.conv3:
            Y = self.bn3(self.conv4(Y))
            X = self.bn_(self.conv3(X))
        return nn.LeakyReLU(0.1)(Y + X)


def residual_block_conv(in_channels, out_channels, num_residuals, batch_norm=True, size_layer_norm=None):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0:
            blk.add_module('residual_{}'.format(i),
                           Residual_conv(in_channels, out_channels, batch_norm, size_layer_norm))
        else:
            blk.add_module('residual_{}'.format(i),
                           Residual_conv(out_channels, out_channels, batch_norm, size_layer_norm))
    return blk


class Classifier(nn.Module):
    def __init__(self, in_channels):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(self.in_channels, 64, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # 64 x 64 x 64
            residual_block_conv(64, 64, 2),
            nn.AvgPool2d(2),
            # 64 x 32 x32
            residual_block_conv(64, 64, 4),
            nn.AvgPool2d(2),
            # 64 x 16 x16
            residual_block_conv(64, 64, 2),
            nn.AvgPool2d(2),
            # 64 x 8 x 8
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 40),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, z_dim, n_classes):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + n_classes, 64 * 32 * 32)
        self.net = nn.Sequential(
            residual_block_conv(64, 64, 2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            residual_block_conv(64, 64, 4),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            residual_block_conv(64, 64, 2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x, labels):
        labels = labels.unsqueeze(2).unsqueeze(3)  # --> N, n_classes, 1 , 1
        x = torch.cat([x, labels], dim=1)
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x).reshape((-1, 64, 32, 32))
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LayerNorm((64, 64, 64)),
            nn.LeakyReLU(0.1),
            residual_block_conv(64, 64, 2, batch_norm=False, size_layer_norm=64),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),
            residual_block_conv(64, 64, 4, batch_norm=False, size_layer_norm=32),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),
            residual_block_conv(64, 64, 2, batch_norm=False, size_layer_norm=16),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)
