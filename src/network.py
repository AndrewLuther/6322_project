import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool

from class_var import DEVICE

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
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.final_conv(x)
        # Since depending on the number of examplars we give
        # We can have more than one examplar here on the batch dimension
        # So we max them to get a single result
        # Max was chosen but can also use mean
        x, _ = torch.max(x, dim=0, keepdim=True) 
        return x
    
class FeatureCorrelationModule(nn.Module):
    def __init__(self):
        super(FeatureCorrelationModule, self).__init__()

    def forward(self, image_feature, examplar_feature):
        # ref : https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        # padding = 3 preserves the size

        # Was getting "inf" values in the correlation_map
        # Going to normalize to avoid inf/nan from large values
        examplar_feature = F.normalize(examplar_feature, p=2, dim=1)

        correlation_map = F.conv2d(image_feature, examplar_feature, stride=1, padding=3)
        return correlation_map
    
class ROIPool(nn.Module):
    def __init__(self, output_size=(7, 7), scales=[0.9, 1.0, 1.1]):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.scales = scales

        # # dummy parameter to get the model's device from (ref: https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently)
        # self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, feature_map, bboxes):
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html

        batch_indices = torch.arange(len(bboxes)).unsqueeze(1).to(DEVICE)  # Shape (B, 1)

        # Concatenate the indices to the first column of bboxes
        # this gives shape [B, 5] or [K, 5] in the docs, where K is number of elements
        # This is required as read here : https://pytorch.org/vision/main/_modules/torchvision/ops/roi_pool.html#roi_pool
        rois = torch.cat([batch_indices, bboxes], dim=1) 

        pooled_outputs = []
        for scale in self.scales:
            pooled_output = roi_pool(
                feature_map,
                boxes=rois,
                output_size=self.output_size,
                spatial_scale=scale
            )
            pooled_outputs.append(pooled_output) 
        
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
        with torch.no_grad():
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
        self.density_prediction = DensityPredictionModule(in_channels=6, out_channels=64)  # 6 from concatenating 3+3 maps

    def forward(self, x, bboxes):
        bboxes = bboxes.squeeze(0)
        # Extract features from both scales
        f_map1, f_map2 = self.feature_extraction(x)
        target_size = f_map1.shape[2:]

        f_maps = [f_map1, f_map2]
        c_maps = []

        for f_map in f_maps:
            scaleW = f_map.shape[2] / x.shape[2] # x scale
            scaleH = f_map.shape[3] / x.shape[3] # y scale
            scaled_bboxes = self._scale_bboxes(bboxes, scaleW, scaleH, f_map.shape[2], f_map.shape[3])
            exemplar_fs = self.roi_pool(f_map, scaled_bboxes)

            # for each of the scaled features (scales of 0.9, 1.0, 1.1)
            for exemplar_f in exemplar_fs:
                c_map = self.feature_correlation(f_map, exemplar_f)
                c_map = F.interpolate(c_map, size=target_size, mode='bilinear', align_corners=False)
                c_maps.append(c_map)
        
        # We want to concatenate the correlation maps such that
        # we have 6 channels, 3 channels for each of the scales, and then for each of the
        # feature maps from the third and fourth blocks  
        c_maps = torch.cat(c_maps, dim=0).permute(1, 0, 2, 3) # shpae [B, 6, H, W]
        return self.density_prediction(c_maps)
    
    def _scale_bboxes(self, bboxes, scaleW, scaleH, H, W):
        scaled_bboxes = bboxes.clone()

        # Clamp to ensure bounding boxes are within image dimensions
        scaled_bboxes[..., 0] = (scaled_bboxes[..., 0] * scaleW).clamp(0, W - 1)  # x1
        scaled_bboxes[..., 1] = (scaled_bboxes[..., 1] * scaleH).clamp(0, H - 1)  # y1
        scaled_bboxes[..., 2] = (scaled_bboxes[..., 2] * scaleW).clamp(0, W - 1)  # x2
        scaled_bboxes[..., 3] = (scaled_bboxes[..., 3] * scaleH).clamp(0, H - 1)  # y2

        return scaled_bboxes





    

