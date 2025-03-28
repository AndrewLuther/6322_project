import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool

class DensityPredictionModule(nn.Module):
    def __init__(self, in_channels):
        super(DensityPredictionModule, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 196, kernel_size=7, padding=3), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(196, 128, kernel_size=5, padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.mean(x, dim=0, keepdim=True) 
        return x
    
class FeatureCorrelationModule(nn.Module):
    def __init__(self):
        super(FeatureCorrelationModule, self).__init__()

    def forward(self, image_feature, examplar_feature):
        kernel_height = examplar_feature.shape[2]
        kernel_width = examplar_feature.shape[3]

        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        padded_image_feature = F.pad(image_feature, (pad_w, pad_w, pad_h, pad_h))
        correlation_map = F.conv2d(padded_image_feature, examplar_feature, stride=1)
        return correlation_map
    
class ROIPool(nn.Module):
    def __init__(self):
        super(ROIPool, self).__init__()

    def forward(self, feature_map, bboxes, output_size, spatial_scale):
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
        # ref: https://pytorch.org/vision/main/generated/torchvision.ops.roi_pool.html
        # bboxes shape is [N, 4]
        pooled_output = roi_pool(
            feature_map,
            boxes=[bboxes], # expects a list of tensors of shape [N, 4]
            output_size=output_size,
            spatial_scale=spatial_scale
        )
        
        return pooled_output

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.eval()
        self.block1 = nn.Sequential(*list(resnet.children())[:4])  # conv1 + bn1 + relu + maxpool
        self.block2 = list(resnet.children())[4]  # Layer 1
        self.block3 = list(resnet.children())[5]  # Layer 2
        self.block4 = list(resnet.children())[6]  # Layer 3

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        f_map1 = self.block3(x)
        f_map2 = self.block4(f_map1)
        return f_map1, f_map2

class FamNet(nn.Module):
    def __init__(self):
        super(FamNet, self).__init__()
        self.feature_extraction = FeatureExtractionModule()
        self.roi_pool = ROIPool()
        self.feature_correlation = FeatureCorrelationModule()
        self.density_prediction = DensityPredictionModule(in_channels=6)

    def forward(self, x, bboxes):
        with torch.no_grad():
            bboxes = bboxes.squeeze(0)
            f_map1, f_map2 = self.feature_extraction(x)
            target_size = f_map1.shape[2:]
            f_maps = [f_map1, f_map2]
            c_maps = []
        
            for f_map in f_maps:
                spatial_scale = f_map.shape[2] / x.shape[2]
                
                scaled_bboxes = bboxes * spatial_scale
                max_width = (scaled_bboxes[:, 2] -  scaled_bboxes[:, 0]).max().item()
                max_height = (scaled_bboxes[:, 3] -  scaled_bboxes[:, 1]).max().item()         
                output_size = (max(int(max_height), 1), max(int(max_width), 1))
                
                # ROI Pooling
                exemplar_fs = []
                exemplar_f = self.roi_pool(f_map, bboxes, output_size, spatial_scale)
                exemplar_fs.append(exemplar_f)
                
                # Loop through scale factors (0.9 and 1.1)
                h, w = exemplar_f.shape[2], exemplar_f.shape[3]
                for scale in [0.9, 1.1]:
                    h_scale = min(f_map.shape[2], round(h * scale))
                    w_scale = min(f_map.shape[3], round(w * scale))
                    exemplar_f_scaled = F.interpolate(exemplar_f, size=(h_scale, w_scale), mode='bilinear', align_corners=False)
                    exemplar_fs.append(exemplar_f_scaled)
                    
                for exemplar_f in exemplar_fs:
                    c_map = self.feature_correlation(f_map, exemplar_f)
                    c_map = F.interpolate(c_map, size=target_size, mode='bilinear', align_corners=False)
                    c_maps.append(c_map)

        c_maps = torch.cat(c_maps, dim=0)
        c_maps = c_maps.permute(1, 0, 2, 3)
        c_maps.requires_grad = True
        return self.density_prediction(c_maps)