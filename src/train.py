import torch
import torch.optim as optim
import torch.nn.functional as F

import datetime

from data import Dataset_Creator, display_sample, display_prediction, save_prediction
from network import FamNet

from class_var import DEVICE

def train_FamNet(num_epochs=1, batch_limit=None, learning_rate=1e-5):
    """
    Train the FamNet model on the dataset.
    """
    # Create the dataset and dataloader
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    # Initialize the model
    model = FamNet().to(DEVICE)
    model.train()  # Set the model to training mode

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
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Print loss every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.12f}")

            if batch_limit != None and batch_idx == batch_limit:
                break

        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}")

        # Save the last prediction
        save_prediction(train_images, train_dmaps, pred_dmaps, "../predictions/test.png")
        save_model(model)

def save_model(model):
    # Save the model to be used for testing
    # ref: https://www.w3schools.com/python/python_datetime.asp
    time = datetime.datetime.now()
    time = time.strftime("%b_%d_%H_%M_%S")
    torch.save(model.state_dict(), f"../saved_models/{time}.pth")

def train_FamNet_single_sample(num_epochs=20, learning_rate=1e-5):
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
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}")
            save_prediction(train_images, train_dmaps, pred_dmaps, "../predictions/test.png")

if __name__ == "__main__":
    #train_FamNet_single_sample(learning_rate=1e-5 ,num_epochs=50)
    train_FamNet(batch_limit=500)