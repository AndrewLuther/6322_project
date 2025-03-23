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

    # TODO
    @staticmethod
    def min_count_loss(density_map_label, density_map_prediction):
        pass


