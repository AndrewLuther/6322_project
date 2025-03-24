import torch
import torch.optim as optim
import torch.nn.functional as F
from data import Dataset_Creator, display_sample, display_prediction
from network import FamNet

def train_FamNet(num_epochs=1, learning_rate=1e-5):
    """
    Train the FamNet model on the dataset.
    """
    # Create the dataset and dataloader
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    # Initialize the model
    model = FamNet()
    model.train()  # Set the model to training mode

    # Loss function and optimizer
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(train_loader):
            
            # Prepare the data (move to device if using CUDA)
            train_images = train_images.cuda() if torch.cuda.is_available() else train_images
            train_dmaps = train_dmaps.cuda() if torch.cuda.is_available() else train_dmaps
            train_bboxes = train_bboxes.cuda() if torch.cuda.is_available() else train_bboxes

            train_dmaps = train_dmaps.unsqueeze(0) # add extra dimension
            optimizer.zero_grad()

            pred_dmaps = torch.sigmoid(model(train_images, train_bboxes))  # Get predicted density maps
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}")
                display_prediction(train_images, train_dmaps, pred_dmaps)

            if batch_idx == 200:
                break

        # Print the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.6f}")
        display_prediction(train_images, train_dmaps, pred_dmaps)

if __name__ == "__main__":
    train_FamNet()
