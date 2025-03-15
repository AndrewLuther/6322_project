import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class DensityPredictionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DensityPredictionModule, self).__init__()

        #5 convolutional blocks (Conv2d + ReLU)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU())

        #1x1 Convolution layer
        self.final_conv = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        # Right now I just gussesd these params, really not sure what to set them too as of now
        # It's supposed to Upsample so I assume a scale_factor=1.0 is wrong, they do mention they scale
        # 0.9 to 1.1 but i couldn't figure out if that was preprocessing step or the upsample step, 
        # I assume preprocessing since 0.9 isn't an upsample? 
        x = F.interpolate(x, scale_factor=1.0, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=1.0, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=1.0, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.final_conv(x)

        return x

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        
        # Load pre-trained ResNet-50 and freeze parameters
        # ref: https://pytorch.org/vision/0.9/models.html
        # ref: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        pass

class FamNet(nn.Module):
    def __init__(self):
        pass 

    def forward(self, x):
        """
        input: x => images input of shape [B, 3 , H, W] 
        """
        pass