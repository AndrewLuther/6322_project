import torch
from util import Util
from data import Dataset_Creator
from accuracy import Accuracy
from test import test_FamNet

import argparse

from device import DEVICE

# Getting the results from table 1

class Eval():

    @staticmethod
    def get_FamNet_accuracies(dataset, model_path="../saved_models/Apr_04_14_19_45.pth", adaptation=False, limit=None):
        pred_counts = test_FamNet(dataset, adaptation=adaptation, limit=limit, model_path=model_path)
        gt_counts = Util.get_ground_truth_counts(dataset, limit=limit)

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
        mean_predictions = mean.repeat(len(dataset)).to(DEVICE)
        return mean_predictions
    

    @staticmethod
    def _get_median_count_for_dataset(dataset=None):
        """
        Returns the median number of objects in the image across the training dataset (unless otherwise specified).
        Used for the median predictor.
        """
        # by default, we want to use the mean of the training dataset, baseline is based on training dataset's median count
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

def eval_with_args():
    # ref: https://stackoverflow.com/questions/16712795/pass-arguments-from-cmd-to-python-script 
    # ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # ref: https://docs.python.org/3/howto/argparse.html 
    parser = argparse.ArgumentParser()

    parser.add_argument('--validation', action=argparse.BooleanOptionalAction, help="test with the validation dataset, instead of the test dataset")
    parser.add_argument('--adaptation', action=argparse.BooleanOptionalAction, help="turns on adaptation loss, which does not improve results for our model")
    parser.add_argument('-l', '--limit', action="store", dest="limit", default=None, help="an option to limit testing to the first n samples of the dataset", type=int)
    parser.add_argument('-m', '--model_path', action="store", dest="model_path", 
                        default="../saved_models/Apr_04_14_19_45.pth", help="the path to the saved model to be evaluated", type=str)
    args = parser.parse_args()

    dataset = Dataset_Creator.get_val_dataset() if args.validation else Dataset_Creator.get_test_dataset()

    mae, rmse = Eval.get_FamNet_accuracies(dataset, model_path=args.model_path, adaptation=args.adaptation, limit=args.limit)
    print(f"MAE:{mae} | RMSE:{rmse}")

if __name__ == "__main__":
    
    # test_dataset = Dataset_Creator.get_test_dataset()
    # # Used this to make sure method of accuracy prediction was correct
    # #print(Experiment1.get_mean_predictor_accuracies(test_dataset))

    # mae, rmse = Eval.get_FamNet_accuracies(test_dataset, adaptation=False, limit=3)
    # print(f"MAE:{mae} | RMSE:{rmse}")

    eval_with_args()






