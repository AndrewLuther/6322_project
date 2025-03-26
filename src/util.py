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
    def get_ground_truth_counts(dataset):
        """
        Get the ground truth counts from the dataset, which are the number of objects in each
        image in the dataset
        """
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        gt_counts = []
        for batch_idx, (images, dmaps, bboxes) in enumerate(dataset_loader):
            gt_counts.append(torch.round(torch.sum(dmaps)))

        gt_counts = torch.stack(gt_counts)

        return gt_counts
