import torch
import numpy as np
import skimage

from util import Util

# Loss calculations used for test-time adaptation (*** Only used at test time, not during training)
class Loss():
    @staticmethod
    def _min_count_loss(density_map, bboxes):
        cropped_density_maps = Util.get_examples_from_bboxes(density_map, bboxes)
        loss = 0
        # total loss is sum of loss for each cropped density map
        for crop in cropped_density_maps:
            # we want the sum of the pixels to be at least 1
            loss += np.max([0, 1 - torch.sum(crop)])
        return loss
            
    @staticmethod
    def _perturbation_loss(density_map, bboxes):
        cropped_density_maps = Util.get_examples_from_bboxes(density_map, bboxes)
        loss = 0
        for crop in cropped_density_maps: 
            dimensions = crop.shape[1:3]
            std_dev = dimensions[0]/4 # quarter of window size?
            gaussian_window = skimage.filters.window(('gaussian', std_dev), dimensions)
            loss += torch.sum((crop - gaussian_window)**2)
        return loss

    @staticmethod
    def adaptation_loss(density_map, bboxes, lambda1=10e-9, lambda2=10e-4):
        return lambda1*Loss.min_count_loss(density_map, bboxes) + lambda2*Loss.perturbation_loss(density_map, bboxes)