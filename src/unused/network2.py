import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool

import math

from debug import save_image

## Code Andrew wrote when debugging, by going through everything in our model architecture and testing one
## line at a time. This code is not used.

class FamNet(nn.Module):
    def __init__(self):
        super(FamNet, self).__init__()
        self.feature_extraction = FeatureExtractionModule()
        self.roi = ROIPool()
        self.correlation = FeatureCorrelationModule()
        self.density_prediction = DensityPredictionModule(in_channels=3)

    def forward(self, x, bboxes):
        with torch.no_grad():
            f_map1, f_map2 = self.feature_extraction(x)
            f_maps = [f_map1, f_map2]

            # 2 maps x 3 scales = 6 correlation maps
            correlation_maps = []

            target_size = f_map1.shape[2:]

            # 2 maps
            for f_map in f_maps:
                scaled_exemplar_features = []
                
                spatial_scale = f_map.shape[2] / x.shape[2]

                # ROI Pool feature map and bboxes to get exemplar features
                exemplar_features = self.roi(f_map, bboxes, spatial_scale)

                # original scale
                scaled_exemplar_features.append(exemplar_features)

                for output_scale in [0.9, 1.1]:
                    scaled_exemplar_features.append(self._scale_exemplar_features(exemplar_features, output_scale))

                # 3 scales
                for scaled_exemplar in scaled_exemplar_features:
                    correlation_map = self.correlation(f_map, scaled_exemplar)

                    # normalize and scale to a consisent target size
                    correlation_map = torch.nn.functional.normalize(correlation_map)
                    correlation_map = F.interpolate(correlation_map, size=target_size, mode='bilinear', align_corners=False)

                    correlation_maps.append(correlation_map)

        correlation_maps = torch.cat(correlation_maps, dim=0)
        correlation_maps.requires_grad = True
        #save_image(correlation_maps[1][0], "display/correlation.png", three_dim=True)

        return self.density_prediction(correlation_maps)
    
    def _scale_exemplar_features(self, exemplar_features, scale):
        # math.ceil to avoid 0
        h, w = exemplar_features.shape[2], exemplar_features.shape[3]
        h_scale = min(h, math.ceil(h * scale)) 
        w_scale = min(w, math.ceil(w * scale))
        return F.interpolate(exemplar_features, size=(h_scale, w_scale), mode='bilinear', align_corners=False)


class ROIPool(nn.Module):
    def __init__(self):
        super(ROIPool, self).__init__()

    def forward(self, f_map, bboxes, spatial_scale):

        bboxes = bboxes.squeeze(0) # get rid of batch dimension

        scaled_bboxes = bboxes * spatial_scale
        max_width = (scaled_bboxes[:, 2] -  scaled_bboxes[:, 0]).max().item()
        max_height = (scaled_bboxes[:, 3] -  scaled_bboxes[:, 1]).max().item()         
        output_size = (math.ceil(max_height), math.ceil(max_width))
        
        pooled_output = roi_pool(
            f_map,
            boxes=[bboxes], # expects a list of tensors of shape [N, 4]
            output_size = output_size,
            spatial_scale=spatial_scale
        )
        
        return pooled_output

class FeatureCorrelationModule(nn.Module):
    def __init__(self):
        super(FeatureCorrelationModule, self).__init__()

    def forward(self, f_map, examplar_f):
        correlation_map = F.conv2d(f_map, examplar_f, stride=1)
        return correlation_map

class FeatureExtractionModule(nn.Module):
    def __init__(self):
        super(FeatureExtractionModule, self).__init__()
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=self.weights)
        self.block1 = nn.Sequential(*list(self.resnet.children())[:4])  # conv1 + bn1 + relu + maxpool
        self.block2 = list(self.resnet.children())[4]  # Layer 1
        self.block3 = list(self.resnet.children())[5]  # Layer 2
        self.block4 = list(self.resnet.children())[6]  # Layer 3

    def forward(self, x):
        self.block1.train(False)
        self.block2.train(False)
        self.block3.train(False)
        self.block4.train(False)

        #x_scaled = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # preprocess = self.weights.transforms()
        # x_preprocessed = preprocess(x)

        #save_image(x_scaled[0], "display/preprocess.png", three_dim=True)

        x = self.block1(x)
        x = self.block2(x)
        f_map1 = self.block3(x)
        f_map2 = self.block4(f_map1)

        f_map1 = torch.nn.functional.normalize(f_map1)
        f_map2 = torch.nn.functional.normalize(f_map2)

        save_image(f_map2[0][0], "display/featureMap.png")

        return f_map1, f_map2
    

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

