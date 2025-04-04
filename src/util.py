import numpy as np
import torch

import datetime

from device import DEVICE

# This file is for any misc data processing functions needed for altering the data inputs/outputs

class Util():
    @staticmethod
    def get_examples_from_bboxes(image, bboxes):
        examples = []
        for bbox in bboxes:
            # tensor is [channels, height, width] | bbox is [x1,y1,x2,y2]
            # add 1 because slice operators don't include larger index
            example = image[:, int(bbox[1]):int(bbox[3])+1, int(bbox[0]):int(bbox[2])+1]
            examples.append(example)
        return examples
    
    @staticmethod
    def get_ground_truth_counts(dataset, limit=None):
        """
        Get the ground truth counts from the dataset, which are the number of objects in each
        image in the dataset
        """
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        gt_counts = []
        for batch_idx, (images, dmaps, bboxes) in enumerate(dataset_loader):
            gt_counts.append(torch.round(torch.sum(dmaps)))
            if limit != None:
                if batch_idx == limit-1: break

        gt_counts = torch.stack(gt_counts).to(DEVICE)

        return gt_counts
    
    @staticmethod
    def save_model(model):
        """
        Saves the trianed model to the saved_models folder so it can be used for evaluation.
        """
        # 
        # ref: https://www.w3schools.com/python/python_datetime.asp
        time = datetime.datetime.now()
        time = time.strftime("%b_%d_%H_%M_%S")
        torch.save(model.state_dict(), f"../saved_models/{time}.pth")
