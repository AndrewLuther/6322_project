import torch
import numpy as np

class Accuracy():
    """
    Holds static methods for returning the MAE or RMSE, given a torch tensor of the ground truth counts for the
    dataset and the predicted counts for the same dataset. 
    """
    @staticmethod
    def get_MAE(ground_truth_counts, pred_counts):
        """
        Return the MAE given the tensor of ground truth counts and the tensor of predicted counts for the dataset

        :param pytorch tensor ground_truth_counts: a tensor of the ground truth counts for the dataset
        :param pytorch tensor pred_counts: a tensor of the predicted counts for the dataset
        """
        return torch.sum(torch.abs(ground_truth_counts - pred_counts))/len(pred_counts)
    
    @staticmethod
    def get_RMSE(ground_truth_counts, pred_counts):
        """
        Return the RMSE given the tensor of ground truth counts and the tensor of predicted counts for the dataset

        :param pytorch tensor ground_truth_counts: a tensor of the ground truth counts for the dataset
        :param pytorch tensor pred_counts: a tensor of the predicted counts for the dataset
        """
        return torch.sqrt(torch.sum((ground_truth_counts - pred_counts)**2)/len(pred_counts))