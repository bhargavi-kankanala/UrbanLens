import torch
import torch.nn as nn


# -------------------------
# Double Convolution Block
# -------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        x1 = self.conv1(x)
        p1 = self.pool(x1)

        x2 = self.conv2(p1)
        p2 = self.pool(x2)

        x3 = self.conv3(p2)
        p3 = self.pool(x3)

        x4 = self.conv4(p3)

        return x1, x2, x3, x4


# -------------------------
# Siamese Encoder
# -------------------------
class SiameseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Same encoder is shared
        # between Time 1 and Time 2
        self.encoder = Encoder()

    def forward(self, image_A, image_B):

        A_features = self.encoder(image_A)
        B_features = self.encoder(image_B)

        differences = []

        for A, B in zip(A_features, B_features):

            differences.append(
                torch.abs(A - B)
            )

        return differences


# -------------------------
# U-Net Up Block
# -------------------------
class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.up1 = UpBlock(
            512,
            256,
            256
        )

        self.up2 = UpBlock(
            256,
            128,
            128
        )

        self.up3 = UpBlock(
            128,
            64,
            64
        )

        self.final_conv = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, differences):

        d1, d2, d3, d4 = differences

        x = self.up1(d4, d3)
        x = self.up2(x, d2)
        x = self.up3(x, d1)

        x = self.final_conv(x)

        return x


# -------------------------
# Complete Siamese U-Net
# -------------------------
class SiameseUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.siamese_encoder = SiameseEncoder()
        self.decoder = Decoder()

    def forward(self, image_A, image_B):

        differences = self.siamese_encoder(
            image_A,
            image_B
        )

        output = self.decoder(
            differences
        )

        return output