import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import cv2  # Or other image processing libraries


class HalfInstanceNorm(nn.Module):
    def __init__(self, in_channels):
        super(HalfInstanceNorm, self).__init__()
        self.half_channels = in_channels // 2
        self.instance_norm = nn.InstanceNorm2d(self.half_channels, affine=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x1, x2 = x[:, :self.half_channels], x[:, self.half_channels:]

        x1 = x1.view(b, self.half_channels, h * w)
        x1 -= x1.min(dim=2, keepdim=True)[0]
        x1 /= x1.max(dim=2, keepdim=True)[0]
        x1 = x1.view(b, self.half_channels, h, w)

        x1 = self.instance_norm(x1)
        x = torch.cat((x1, x2), dim=1)
        return x


class FeatureConversionModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureConversionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        # 添加半实例归一化
        self.half_instance_norm = HalfInstanceNorm(in_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.half_instance_norm(x)
        x = self.gelu(x)
        x += residual  # Residual connection
        return x

# Define the U-Net Generator with weight standardization
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        # 编码器路径（包含5个下采样层）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 解码器路径（包含5个上采样层）
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

        # 在编码器-解码器结构中集成的九个FCM
        self.fcm1 = FeatureConversionModule(64)
        self.fcm2 = FeatureConversionModule(128)
        self.fcm3 = FeatureConversionModule(256)
        self.fcm4 = FeatureConversionModule(512)
        self.fcm5 = FeatureConversionModule(1024)
        self.fcm6 = FeatureConversionModule(512)
        self.fcm7 = FeatureConversionModule(256)
        self.fcm8 = FeatureConversionModule(128)
        self.fcm9 = FeatureConversionModule(64)

    def forward(self, x):
        # Encoder path with FCMs
        enc1 = self.encoder1(x)
        fcm1 = self.fcm1(enc1)

        enc2 = self.encoder2(fcm1)
        fcm2 = self.fcm2(enc2)

        enc3 = self.encoder3(fcm2)
        fcm3 = self.fcm3(enc3)

        enc4 = self.encoder4(fcm3)
        fcm4 = self.fcm4(enc4)

        enc5 = self.encoder5(fcm4)
        fcm5 = self.fcm5(enc5)

        # Decoder path with FCMs
        dec1 = self.decoder1(fcm5)
        cat1 = torch.cat((dec1, fcm4), dim=1)  # Concatenate with the corresponding FCM
        fcm6 = self.fcm6(cat1)

        dec2 = self.decoder2(fcm6)
        cat2 = torch.cat((dec2, fcm3), dim=1)
        fcm7 = self.fcm7(cat2)

        dec3 = self.decoder3(fcm7)
        cat3 = torch.cat((dec3, fcm2), dim=1)
        fcm8 = self.fcm8(cat3)

        dec4 = self.decoder4(fcm8)
        cat4 = torch.cat((dec4, fcm1), dim=1)
        fcm9 = self.fcm9(cat4)

        # Output derained image
        derained_image = fcm9
        return derained_image

# Build the Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # Four downsampling layers with even-sized kernels, symmetric padding, instance normalization, and LeakyReLU
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # Final layer for real/fake prediction
        self.final = nn.Conv2d(512, 1, 4, stride=2, padding=1, padding_mode='reflect')

    def forward(self, x):
        # Pass the input image through the downsampling layers
        down1_out = self.down1(x)
        down2_out = self.down2(down1_out)
        down3_out = self.down3(down2_out)
        down4_out = self.down4(down3_out)

        # Final layer for real/fake prediction
        logits = self.final(down4_out)

        return logits


# Remember to adjust kernel sizes, padding, and activation functions based on the specific details in your instructions

#Define Loss Functions:
# L1 loss
def l1_loss(fake_image, gt_image):
    return torch.nn.functional.l1_loss(fake_image, gt_image)

# High-level semantic loss



# lgan_loss 是 log(1 - D(G(Irain))) 项
def lgan_loss(fake_output):
    return -torch.log(1 - fake_output)


def hls_loss(fake_image, gt_image, vgg_model):
    # 使用提供的 VGG 模型从两个图像中提取特征
    fake_features = vgg_model(fake_image)
    gt_features = vgg_model(gt_image)

    # LapStyle（LS）损失
    lambda_ls = [1, 1]  # 根据需要调整每个层的权重
    ls_loss = 0
    for i in range(3, 5):
        ls_loss += lambda_ls[i - 3] * F.l1_loss(fake_features[i], gt_features[i])

    return ls_loss


# 组合生成器损失
def generator_loss(fake_image, gt_image, fake_output, vgg_model):
    l1_loss = F.l1_loss(fake_image, gt_image)
    lambda_g = 0.5  # 根据需要调整 GAN 损失的权重
    gan_loss = lambda_g * lgan_loss(fake_output)
    ls_loss = hls_loss(fake_image, gt_image, vgg_model)

    total_loss = l1_loss + gan_loss + ls_loss
    return total_loss


# Discriminator loss
def discriminator_loss(fake_image, gt_image):
    fake_pred = discriminator(fake_image)
    real_pred = discriminator(gt_image)
    # Use binary cross-entropy loss
    return torch.nn.functional.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred)) + \
        torch.nn.functional.binary_cross_entropy(real_pred, torch.ones_like(real_pred))

#Training:
# Define optimizers for generator and discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Train the network in a loop:
for epoch in range(100):
    for i in range(10):
        # Get a batch of rainy and ground truth images
        rainy_image, gt_image = ...
        # Forward pass and calculate losses
        fake_image = generator(rainy_image)
        g_loss = generator_loss(fake_image, gt_image)
        d_loss = discriminator_loss(fake_image, gt_image)
        # Backward pass and update parameters
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G

#Set up data loading and preprocessing functions:
def load_data(data_path):
    # Load rainy and ground truth images from the dataset
    # Preprocess images (e.g., normalize, resize)
    return rainy_images, ground_truth_images

#Define training loop:
# Create instances of generator and discriminator networks
generator = UNetGenerator(...)
discriminator = Discriminator(...)

# Define optimizers for both networks
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Load dataset
rainy_images, gt_images = load_data(data_path)

# Training loop
for epoch in range(num_epochs):
    # Iterate through batches of data
    for i in range(num_batches):
        # Forward pass, calculate losses, backpropagate, and update parameters (as shown in the examples)
        # Get a batch of rainy and ground truth images
        rainy_image, gt_image = rainy_images[i * batch_size:(i + 1) * batch_size], gt_images[i * batch_size:(i + 1) * batch_size]

        # Forward pass for generator
        fake_image = generator(rainy_image)

        # Discriminator forward pass
        fake_output = discriminator(fake_image.detach())  # Detach to avoid gradient flow
        real_output = discriminator(gt_image)

        # Calculate losses
        g_loss = generator_loss(fake_image, gt_image, fake_output, vgg_model)
        d_loss = discriminator_loss(fake_output, real_output)

        # Backward pass and update parameters for discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Backward pass and update parameters for generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
#Define rain removal function:
        def remove_rain(rainy_image):
            # Preprocess input image
            # Pass through trained generator
            # Postprocess output image
            return derained_image

#Load pre-trained VGG19 model (for high-level semantic loss):
vgg19 = torchvision.models.vgg19(pretrained=True).features
