import numpy as np
import torch

# This file is for any misc data processing functions needed for altering the data inputs/outputs

class Util():
    @staticmethod
    def get_examples_from_bboxes(image, bboxes):
        examples = []
        for bbox in bboxes:
            # add 1 because slice operators don't include larger index
            example = image[:, int(bbox[0]):int(bbox[2])+1, int(bbox[1]):int(bbox[3])+1]
            examples.append(example)
        return examples

    @staticmethod
    def min_count_loss(density_map, bboxes):
        cropped_density_maps = Util.get_examples_from_bboxes(density_map, bboxes)
        loss = 0
        for crop in cropped_density_maps:
            loss += np.max([0, 1 - torch.sum(crop)])
        return loss
            
    #TODO
    @staticmethod
    def perturbation_loss():
        pass

    #TODO
    @staticmethod
    def adaptation_loss():
        pass

