import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool

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
    
class FeatureCorrelationModule(nn.Module):
    def __init__(self):
        super(FeatureCorrelationModule, self).__init__()

    def forward(self, f_map, scaled_f_map):
        # f_map: torch.Size([1, 512, 48, 86])
        # scaled_f_map:  torch.Size([3, 512, 7, 7])
        # ref : https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        correlation_map = F.conv2d(f_map, scaled_f_map, stride=1, padding=0)

        return correlation_map
    
class ROIPool(nn.Module):
    def __init__(self, output_size=(7, 7), scales=[0.9, 1.0, 1.1]):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.scales = scales

    def forward(self, feature_map, bboxes):
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html

        indices = torch.arange(len(bboxes)).unsqueeze(1)  # Shape (B, 1)

        # Concatenate the indices to the second column of bboxes
        # this gives shape [B, 5] or [K, 5] in the docs, where K is number of elements
        bboxes = torch.cat([bboxes[:, :1], indices, bboxes[:, 1:]], dim=1)

        pooled_outputs = []
        for scale in self.scales:
            pooled_output = roi_pool(
                feature_map,
                boxes=bboxes,
                output_size=self.output_size,
                spatial_scale=scale
            )
            pooled_outputs.append(pooled_output)
        
        pooled_outputs = torch.cat(pooled_outputs, dim=0)
        return pooled_outputs

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
        return f_map1, f_map2

class FamNet(nn.Module):
    def __init__(self):
        super(FamNet, self).__init__()
        self.feature_extraction = FeatureExtractionModule()
        self.roi_pool = ROIPool()
        self.feature_correlation = FeatureCorrelationModule()
        self.density_prediction = DensityPredictionModule(in_channels=3, out_channels=64)

    def forward(self, x, bboxes):
        f_map1, f_map2 = self.feature_extraction(x)

        # Compute spatial scale as (feature map size / image size)
        # using height here to compute scale
        scale1 =  f_map1.shape[2] / x.shape[2]  # gives 0.125,
        scale2 =  f_map2.shape[2] / x.shape[2] # gives 0.0625

        bboxes_1 = self._scale_bboxes(bboxes.clone(), scale=scale1)
        bboxes_2 = self._scale_bboxes(bboxes.clone(), scale=scale2)

        scaled_f_map1 = self.roi_pool(f_map1, bboxes_1)
        scaled_f_map2 = self.roi_pool(f_map2, bboxes_2)

        c_map_1 = self.feature_correlation(f_map1, scaled_f_map1)
        c_map_2 = self.feature_correlation(f_map2, scaled_f_map2)
        
        # from the paper it says: 
        # "The correlation maps are concatenated and fed into the density prediction module."
        # howerver they are not the same size, c_map_2 is smaller than c_map_1, therefor
        # need to upsample c_map_2.
        c_map_2 = F.interpolate(c_map_2, size=c_map_1.shape[2:], mode='bilinear', align_corners=False)
        d_map_input = torch.cat((c_map_1, c_map_2), dim=0) # torch.Size([2, 3, 42, 45])

        return self.density_prediction(d_map_input)
       
    def _scale_bboxes(self, bboxes, scale = 1):
        """
        Scales bounding boxe around their center by a given scale factor.
        """
        x1, y1 = bboxes[:, 0], bboxes[:, 1]
        x2, y2 = bboxes[:, 2], bboxes[:, 3]

        # Compute center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Scale width and height
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale

        # Reconstruct scaled box
        scaled_x1 = cx - w / 2
        scaled_y1 = cy - h / 2
        scaled_x2 = cx + w / 2
        scaled_y2 = cy + h / 2

        # Stack back into [K, 4]
        return torch.stack([scaled_x1, scaled_y1, scaled_x2, scaled_y2], dim=1)

    

