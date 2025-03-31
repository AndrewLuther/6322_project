import math
import torch
import torch.nn as nn
import torch.nn.init as init
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

    def forward(self, f_map, examplar_f):
        correlation_map = F.conv2d(f_map, examplar_f, stride=1)
        return correlation_map
    
class ROIPool(nn.Module):
    def __init__(self):
        super(ROIPool, self).__init__()

    def forward(self, f_map, bboxes, spatial_scale):
        scaled_bboxes = bboxes * spatial_scale
        max_width = (scaled_bboxes[:, 2] -  scaled_bboxes[:, 0]).max().item()
        max_height = (scaled_bboxes[:, 3] -  scaled_bboxes[:, 1]).max().item()         
        output_size = (math.ceil(max_height), math.ceil(max_width))
        
        pooled_output = roi_pool(
            f_map,
            boxes=[bboxes], # expects a list of tensors of shape [N, 4]
            output_size=output_size,
            spatial_scale=spatial_scale
        )
        
        return pooled_output

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.train(False)
        self.block1 = nn.Sequential(*list(resnet.children())[:4])  # conv1 + bn1 + relu + maxpool
        self.block2 = list(resnet.children())[4]  # Layer 1
        self.block3 = list(resnet.children())[5]  # Layer 2
        self.block4 = list(resnet.children())[6]  # Layer 3

    def forward(self, x):
        self.block1.train(False)
        self.block2.train(False)
        self.block3.train(False)
        self.block4.train(False)

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

        #self._initialize_density_weights()
    
    def _initialize_density_weights(self):
        # Loop through all layers in the density prediction module and initialize their weights
        for m in self.density_prediction.modules():
            if isinstance(m, nn.Conv2d):  # For convolution layers
                init.normal_(m.weight, mean=0, std=0.05)
                if m.bias is not None:
                    init.zeros_(m.bias)  # Initialize bias to zeros
        
    def forward(self, x, bboxes):
        with torch.no_grad():
            bboxes = bboxes.squeeze(0)
            f_map1, f_map2 = self.feature_extraction(x)
            target_size = f_map1.shape[2:]
            f_maps = [f_map1, f_map2]
            c_maps = []
        
            for f_map in f_maps:
                spatial_scale = f_map.shape[2] / x.shape[2]
                # ROI Pooling
                exemplar_fs = []
                exemplar_f = self.roi_pool(f_map, bboxes, spatial_scale)
                exemplar_fs.append(exemplar_f)
                
                # Loop through scale factors (0.9 and 1.1)
                h, w = exemplar_f.shape[2], exemplar_f.shape[3]
                for scale in [0.9, 1.1]:
                    # math.ceil to avoid 0
                    h_scale = min(h, math.ceil(h * scale)) 
                    w_scale = min(w, math.ceil(w * scale))
                    
                    exemplar_f_scaled = F.interpolate(exemplar_f, size=(h_scale, w_scale), mode='bilinear', align_corners=False)
                    exemplar_fs.append(exemplar_f_scaled)

                for exemplar_f in exemplar_fs:
                    c_map = self.feature_correlation(f_map, exemplar_f)
                    if target_size is None:
                        h, w = c_map.shape[2], c_map.shape[3]
                        target_size = (h, w)
                    else:
                        c_map = F.interpolate(c_map, size=target_size, mode='bilinear', align_corners=False)
                    c_maps.append(c_map)

        c_maps = torch.cat(c_maps, dim=0).permute(1, 0, 2, 3)
        c_maps.requires_grad = True
        return self.density_prediction(c_maps)