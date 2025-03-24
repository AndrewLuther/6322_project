import torch
import numpy as np
import skimage

from util import Util
from data import Dataset_Creator

# Accuracy calculations
class Accuracy():
    @staticmethod
    def get_MAE(ground_truth_counts, pred_counts):
        return torch.sum(torch.abs(ground_truth_counts - pred_counts))/len(pred_counts)
    
    @staticmethod
    def get_RMSE(ground_truth_counts, pred_counts):
        return torch.sqrt(torch.sum((ground_truth_counts - pred_counts)**2)/len(pred_counts))


if __name__ == "__main__":
    val_dataset = Dataset_Creator.get_val_dataset()

    gt_counts = Util.get_ground_truth_counts(val_dataset)
    mean_counts = Util.get_mean_predictor_counts(val_dataset)

    print(Accuracy.get_MAE(gt_counts, mean_counts))
    print(Accuracy.get_RMSE(gt_counts, mean_counts))





