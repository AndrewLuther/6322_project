import torch
from util import Util
from data import Dataset_Creator
from accuracy import Accuracy
from test import test_FamNet

import gc

gc.collect()

# Getting the results from table 1

class Experiment1():

    @staticmethod
    def get_FamNet_accuracies():
        pred_counts = test_FamNet()
        gt_counts = Util.get_ground_truth_counts(Dataset_Creator.get_val_dataset())

        mae = Accuracy.get_MAE(gt_counts, pred_counts)
        rmse = Accuracy.get_RMSE(gt_counts, pred_counts)
        return mae, rmse


    @staticmethod
    def get_mean_predictor_accuracies(dataset):
        gt_counts = Util.get_ground_truth_counts(dataset)
        mean_counts = Mean_Median_Predictor.get_mean_predictor_counts(dataset)

        mae = Accuracy.get_MAE(gt_counts, mean_counts)
        rmse = Accuracy.get_RMSE(gt_counts, mean_counts)
        return mae, rmse
    
    @staticmethod
    def get_median_predictor_accuracies(dataset):
        gt_counts = Util.get_ground_truth_counts(dataset)
        median_counts = Mean_Median_Predictor.get_median_predictor_counts(dataset)

        mae = Accuracy.get_MAE(gt_counts, median_counts)
        rmse = Accuracy.get_RMSE(gt_counts, median_counts)
        return mae, rmse
    

class Mean_Median_Predictor():
    @staticmethod
    def _get_mean_count_for_dataset(dataset=None):
        """
        Returns the average number of objects per image in the whole training dataset (unless otherwise specified), for the
        mean predictor (which can serve as a baseline for results)
        """
        #return torch.tensor([63.5389]) # this is the mean for val dataset in case we don't want to wait for this to compute it

        # by default, we want to use the mean of the training dataset, baseline is based on training dataset's mean count
        if dataset == None:
            dataset = Dataset_Creator.get_training_dataset()

        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        total_objects = 0
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(dataset_loader):
            total_objects += torch.round(torch.sum(train_dmaps))
            
        return total_objects/len(dataset)
    
    @staticmethod
    def get_mean_predictor_counts(dataset):
        """
        Returns the counts for a dataset for a predictor that always predicts the mean
        number of objects in the image. Serves as a baseline predictor to compare to.
        Model should preform better than this.
        """
        mean = Mean_Median_Predictor._get_mean_count_for_dataset()
        mean_predictions = mean.repeat(len(dataset))
        return mean_predictions
    

    @staticmethod
    def _get_median_count_for_dataset(dataset=None):
        """
        Returns the median number of objects in the image across the training dataset (unless otherwise specified).
        Used for the median predictor.
        """
        # by default, we want to use the mean of the training dataset, baseline is based on training dataset's mean count
        if dataset == None:
            dataset = Dataset_Creator.get_training_dataset()

        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        counts = []
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(dataset_loader):
            counts.append(torch.sum(train_dmaps))
        counts = torch.stack(counts)
        return torch.median(counts)
    
    @staticmethod
    def get_median_predictor_counts(dataset):
        """
        Returns the counts for a dataset for a predictor that always predicts the median
        number of objects in the the images across the dataset. Serves as a baseline predictor to
        compare to. The model should perform better than this.
        """
        median = Mean_Median_Predictor._get_median_count_for_dataset()
        median_predictions = median.repeat(len(dataset))
        return median_predictions

if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_dataset = Dataset_Creator.get_test_dataset()
    val_dataset = Dataset_Creator.get_val_dataset()

    #print(Experiment1.get_mean_predictor_accuracies(test_dataset))
    print(Experiment1.get_FamNet_accuracies())






