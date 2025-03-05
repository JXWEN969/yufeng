from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetPlusPlus, self).__init__()

        features = init_features
        self.encoder1 = UNetPlusPlus._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetPlusPlus._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNetPlusPlus._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetPlusPlus._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetPlusPlus._block(features * 8, features * 16, name="bottleneck")

        # Nested skip pathways and deep supervision
        self.conv_x00 = self.encoder1
        self.conv_x10 = self.encoder2
        self.conv_x20 = self.encoder3
        self.conv_x30 = self.encoder4
        self.conv_x40 = self.bottleneck

        self.up_x10 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.conv_x01 = UNetPlusPlus._block(features * 2, features, name="conv_x01")

        self.up_x20 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.conv_x11 = UNetPlusPlus._block(features * 4, features * 2, name="conv_x11")
        self.up_x21 = nn.ConvTranspose2d(features * 4, features, kernel_size=2, stride=2)
        self.conv_x02 = UNetPlusPlus._block(features * 3, features, name="conv_x02")

        self.up_x30 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.conv_x21 = UNetPlusPlus._block(features * 8, features * 4, name="conv_x21")
        self.up_x31 = nn.ConvTranspose2d(features * 8, features * 2, kernel_size=2, stride=2)
        self.conv_x12 = UNetPlusPlus._block(features * 6, features * 2, name="conv_x12")
        self.up_x32 = nn.ConvTranspose2d(features * 8, features, kernel_size=2, stride=2)
        self.conv_x03 = UNetPlusPlus._block(features * 4, features, name="conv_x03")

        self.up_x40 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv_x31 = UNetPlusPlus._block(features * 16, features * 8, name="conv_x31")
        self.up_x41 = nn.ConvTranspose2d(features * 16, features * 4, kernel_size=2, stride=2)
        self.conv_x22 = UNetPlusPlus._block(features * 12, features * 4, name="conv_x22")
        self.up_x42 = nn.ConvTranspose2d(features * 16, features * 2, kernel_size=2, stride=2)
        self.conv_x13 = UNetPlusPlus._block(features * 10, features * 2, name="conv_x13")
        self.up_x43 = nn.ConvTranspose2d(features * 16, features, kernel_size=2, stride=2)
        self.conv_x04 = UNetPlusPlus._block(features * 5, features, name="conv_x04")

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pathway
        x00 = self.conv_x00(x)
        x10 = self.pool1(x00)
        x10 = self.conv_x10(x10)
        x20 = self.pool2(x10)
        x20 = self.conv_x20(x20)
        x30 = self.pool3(x20)
        x30 = self.conv_x30(x30)
        x40 = self.pool4(x30)
        x40 = self.conv_x40(x40)

        # Decoder pathway with nested skip connections
        x01 = self.up_x10(x10)
        x01 = torch.cat((x01, x00), dim=1)
        x01 = self.conv_x01(x01)

        x11 = self.up_x20(x20)
        x11 = torch.cat((x11, x10), dim=1)
        x11 = self.conv_x11(x11)
        x02 = self.up_x21(x11)
        x02 = torch.cat((x02, x01, x00), dim=1)
        x02 = self.conv_x02(x02)

        x21 = self.up_x30(x30)
        x21 = torch.cat((x21, x20), dim=1)
        x21 = self.conv_x21(x21)
        x12 = self.up_x31(x21)
        x12 = torch.cat((x12, x11, x10), dim=1)
        x12 = self.conv_x12(x12)
        x03 = self.up_x32(x12)
        x03 = torch.cat((x03, x02, x01, x00), dim=1)
        x03 = self.conv_x03(x03)

        x31 = self.up_x40(x40)
        x31 = torch.cat((x31, x30), dim=1)
        x31 = self.conv_x31(x31)
        x22 = self.up_x41(x31)
        x22 = torch.cat((x22, x21, x20), dim=1)
        x22 = self.conv_x22(x22)
        x13 = self.up_x42(x22)
        x13 = torch.cat((x13, x12, x11, x10), dim=1)
        x13 = self.conv_x13(x13)
        x04 = self.up_x43(x13)
        x04 = torch.cat((x04, x03, x02, x01, x00), dim=1)
        x04 = self.conv_x04(x04)

        # Final convolution
        output = self.final_conv(x04)
        return torch.sigmoid(output)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True))
                ]
            )
        )


# Instantiate the model
model = UNetPlusPlus(n_channels=3, n_classes=1)

# Print the model summary (optional)
print(model)