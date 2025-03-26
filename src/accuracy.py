import torch
import numpy as np

# Accuracy calculations
class Accuracy():
    @staticmethod
    def get_MAE(ground_truth_counts, pred_counts):
        return torch.sum(torch.abs(ground_truth_counts - pred_counts))/len(pred_counts)
    
    @staticmethod
    def get_RMSE(ground_truth_counts, pred_counts):
        return torch.sqrt(torch.sum((ground_truth_counts - pred_counts)**2)/len(pred_counts))




