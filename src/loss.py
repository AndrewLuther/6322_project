import torch
import numpy as np
import skimage

from util import Util
from debug import save_image
from device import DEVICE

# Loss calculations used for test-time adaptation (*** Only used at test time, not during training)
class Loss():
    @staticmethod
    def _min_count_loss(density_map, bboxes):
        cropped_density_maps = Util.get_examples_from_bboxes(density_map, bboxes)
        loss = torch.tensor([0], dtype=float).to(DEVICE)
        loss.requires_grad_()
        # total loss is sum of loss for each cropped density map
        for crop in cropped_density_maps:
            crop_count = torch.sum(crop)
            # we want the sum of the pixels to be at least 1
            loss = loss + torch.max(torch.tensor([0, 1 - crop_count]))
        return loss
            
    @staticmethod
    def _perturbation_loss(density_map, bboxes):
        cropped_density_maps = Util.get_examples_from_bboxes(density_map, bboxes)
        loss = torch.tensor([0], dtype=float).to(DEVICE)
        loss.requires_grad_()
        for crop in cropped_density_maps:
            crop = crop[0]
            dimensions = crop.shape
            std_dev = crop.shape[0]/4 # quarter of window size?
            gaussian = skimage.filters.window(('gaussian', std_dev), dimensions)
            gaussian = gaussian / np.sum(gaussian) # normalize
            gaussian_window = torch.from_numpy(gaussian).to(DEVICE)

            #save_image(gaussian_window, "display/gaussian.png", tensor2=crop) # can use this to display the gaussian filter and crop
            loss = loss + torch.sum((crop - gaussian_window)**2)
        return loss

    @staticmethod
    def adaptation_loss(density_map, bboxes, lambda1=10e-9, lambda2=10e-4):
        adaptation_loss = lambda1*Loss._min_count_loss(density_map, bboxes) + lambda2*Loss._perturbation_loss(density_map, bboxes)
        return adaptation_loss