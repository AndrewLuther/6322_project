import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from data import Dataset_Creator

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

        # Extract first four blocks of ResNet-50
        # ref: https://github.com/pytorch/vision/blob/95131de394543a7c34bd51932bdfce21dae516c1/torchvision/models/resnet.py#L197-L212
        self.block1 = nn.Sequential(*list(resnet.children())[:4])  # conv1 + bn1 + relu + maxpool1
        self.block2 = list(resnet.children())[4]  # Layer 1
        self.block3 = list(resnet.children())[5]  # Layer 2
        self.block4 = list(resnet.children())[6]  # Layer 3

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        f_map1 = self.block3(x)  # Feature maps from third block
        f_map2 = self.block4(f_map1)  # Feature maps from fourth block

        # TODO Next we need to perform ROI pooling with the bounding boxes of the exemplar objects
        # This will involve this ROI pytorch function which will take f_map1 and f_map2 as input,
        # along with the bounding boxes from the dataloader
        # https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html

        # Then our ROI Pooled exemplar features will be scaled to 0.9, 1.1 and original (I guess that's 1?)

        # Then the output of the scaled ROI pooling and the f_map1 and f_map2 go into our correlation layer, 
        # still trying to figure out what that is...

        # This will give multiple correlation maps, for each different scale, which are concatenated and are then 
        # the output of this module (this is what will then be fed into our density prediction module)

        return f_map1, f_map2

class FamNet(nn.Module):
    def __init__(self):
        pass 

    def forward(self, x):
        """
        input: x => images input of shape [B, 3 , H, W] 
        """
        pass

if __name__ == "__main__":

    # Testing code
    train_data = Dataset_Creator.get_training_dataset()

    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_examples = next(iter(train_loader))

    model = FeatureExtractionModule()
    model.train(False)

    features1, features2 = model(train_images)
    print(features2)
    

