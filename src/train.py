import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse

from data import Dataset_Creator
from debug import display_prediction, display_sample, save_prediction
from network import FamNet as FamNet
from logger import Logger
from util import Util

from device import DEVICE

def train_FamNet(num_epochs=1, batch_limit=None, learning_rate=1e-5, log=True):
    """
    Train the FamNet model on the training dataset.
    """

    # Create the dataset and dataloader
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    # Initialize the model
    model = FamNet().to(DEVICE)
    model.train()  # Set the model to training mode

    # Debugging/logging stuff
    if log:
        logger = Logger()

        for name, module in model.density_prediction.named_modules():
            if isinstance(module, torch.nn.Conv2d):  
                module.register_forward_hook(logger.forward_hook)
                module.register_full_backward_hook(logger.backward_hook)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss().to(DEVICE)  # Mean Squared Error Loss
    optimizer = optim.Adam(model.density_prediction.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(train_loader):
            
            # Prepare the data (move to device if using CUDA)
            train_images = train_images.to(DEVICE)
            train_dmaps = train_dmaps.to(DEVICE)
            train_bboxes = train_bboxes.to(DEVICE)
            train_dmaps = train_dmaps.unsqueeze(0) # add extra dimension

            optimizer.zero_grad()
            
            pred_dmaps = model(train_images, train_bboxes)  # Get predicted density maps
            # Ensure pred_dmaps has same shape as train_dmaps
            # Sometimes pred_dmaps is a few pixels off on the width
            if pred_dmaps.shape != train_dmaps.shape:
                pred_dmaps = F.interpolate(pred_dmaps, size=train_dmaps.shape[2:], mode='bilinear', align_corners=False)
            
            # Compute the loss (MSE between predicted and ground truth density maps)
            loss = criterion(pred_dmaps, train_dmaps)
            epoch_loss += loss.item()
            if log: logger.add_scalar('loss', loss)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Print loss every 10 batches
                if batch_limit:
                    num_batches = batch_limit
                else:
                    num_batches = len(train_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.12f}")

            if batch_limit != None and batch_idx == (batch_limit-1):
                break

            if batch_idx == 0 or batch_idx == 9:
                save_prediction(train_images, train_dmaps, pred_dmaps, f"../predictions/prediction{batch_idx}.png")

            if log: logger.increment()

        if log: logger.finish()
        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}")

        # Save the last prediction
        save_prediction(train_images, train_dmaps, pred_dmaps, "../predictions/final_prediction.png")
        Util.save_model(model)



def train_FamNet_single_sample(num_epochs=50, learning_rate=1e-5):
    """
    Train the FamNet model on a single sample for multiple epochs.
    """

    # Create the dataset and dataloader
    train_data = Dataset_Creator.get_training_dataset()
    single_sample = torch.utils.data.Subset(train_data, [0])  # Use only the first sample
    train_loader = torch.utils.data.DataLoader(single_sample, batch_size=1, shuffle=False)

    # Initialize the model
    model = FamNet().to(DEVICE)
    model.train()  # Set the model to training mode
    model.feature_extraction.eval() 

    # Debugging/logging stuff
    logger = Logger()

    for name, module in model.density_prediction.named_modules():
        if isinstance(module, torch.nn.Conv2d):  
            module.register_forward_hook(logger.forward_hook)
            module.register_full_backward_hook(logger.backward_hook)

    # Loss function and optimizer
    criterion = torch.nn.MSELoss().to(DEVICE)  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop for multiple epochs on a single sample
    for epoch in range(num_epochs):
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(train_loader):
            # Prepare the data (move to device if using CUDA)
            train_images = train_images.to(DEVICE)
            train_dmaps = train_dmaps.to(DEVICE)
            train_bboxes = train_bboxes.to(DEVICE)
            train_dmaps = train_dmaps.unsqueeze(0)  # Add extra dimension

            optimizer.zero_grad()
            pred_dmaps = model(train_images, train_bboxes)  # Get predicted density maps

            # Ensure pred_dmaps has same shape as train_dmaps
            if pred_dmaps.shape != train_dmaps.shape:
                pred_dmaps = F.interpolate(pred_dmaps, size=train_dmaps.shape[2:], mode='bilinear', align_corners=False)

            # Compute the loss (MSE between predicted and ground truth density maps)
            loss = criterion(pred_dmaps, train_dmaps)

            logger.add_scalar('loss', loss)

            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}")

            if(epoch == num_epochs-1):
                save_prediction(train_images, train_dmaps, pred_dmaps, "../predictions/final_prediction.png")

            logger.increment()

    logger.finish()

def train_with_args():
    """
    Trains the model using commandline arguments provided by the user
    """
    # ref: https://stackoverflow.com/questions/16712795/pass-arguments-from-cmd-to-python-script 
    # ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # ref: https://docs.python.org/3/howto/argparse.html 
    parser = argparse.ArgumentParser()

    parser.add_argument('--single', action=argparse.BooleanOptionalAction, help="whether or not to train the model on one sample")
    parser.add_argument('-e', '--num_epochs', action="store", dest="num_epochs", default=None, help="the number of epochs to run the training for", type=int)
    parser.add_argument('-b', '--batch_limit', action="store", dest="batch_limit", default=None, help="batch cutoff to stop the training early", type=int)
    parser.add_argument('-lr', '--learning_rate', action="store", dest="learning_rate", default=1e-5, help="the step size during gradient descent", type=float)
    args = parser.parse_args()

    # note: defaults for num_epochs depend on if single sample or not (with single sample we likely want several epochs)
    if args.single:
        epochs = 50 if args.num_epochs == None else args.num_epochs
        train_FamNet_single_sample(num_epochs=epochs, learning_rate=args.learning_rate)
    else:
        epochs = 1 if args.num_epochs == None else args.num_epochs
        train_FamNet(num_epochs=epochs, batch_limit=args.batch_limit, learning_rate=args.learning_rate)


if __name__ == "__main__":
    train_with_args()
    #train_FamNet(batch_limit=10)