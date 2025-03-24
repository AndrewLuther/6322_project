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
            gt_counts.append(torch.sum(dmaps))

        gt_counts = torch.stack(gt_counts)

        return gt_counts
    
    @staticmethod
    def get_mean_count_for_dataset(dataset):
        """
        Returns the average number of objects per image in the whole dataset, for the
        mean predictor (which can serve as a baseline for results)
        """
        #return torch.tensor([63.5389]) # temp

        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        total_objects = 0
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(dataset_loader):
            total_objects += torch.sum(train_dmaps)
            
        return total_objects/len(dataset)
    
    @staticmethod
    def get_mean_predictor_counts(dataset):
        """
        Returns the counts for a dataset for a predictor that always predicts the mean
        number of objects in the image. Serves as a baseline predictor to compare to.
        Model should preform better than this.
        """
        mean = Util.get_mean_count_for_dataset(dataset)
        mean_predictions = mean.repeat(len(dataset))
        return mean_predictions

