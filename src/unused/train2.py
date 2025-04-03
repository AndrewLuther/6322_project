from device import DEVICE
from data import Dataset_Creator
from debug import save_prediction

from network2 import FamNet as F2

import torch
import torch.optim as optim

## Code Andrew wrote when debugging, by going through everything in our model architecture and testing one
## line at a time. This code is not used.

def train(num_epochs=20, learning_rate=1e-5):
    """
    Rewritten training method on one dataset sample, written while debugging.
    """
    # Create the dataset and dataloader
    train_data = Dataset_Creator.get_training_dataset()
    single_sample = torch.utils.data.Subset(train_data, [1])  # Use only the first sample
    train_loader = torch.utils.data.DataLoader(single_sample, batch_size=1, shuffle=False)

    model = F2().to(DEVICE)
    model.eval()

    # Loss function and optimizer
    criterion = torch.nn.MSELoss().to(DEVICE)  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_idx, (train_images, train_dmaps, train_bboxes) in enumerate(train_loader):
            # Prepare the data (move to device if using CUDA)
            train_images = train_images.to(DEVICE)
            train_dmaps = train_dmaps.to(DEVICE).unsqueeze(0)
            train_bboxes = train_bboxes.to(DEVICE)

            optimizer.zero_grad()
            pred_dmaps = model(train_images, train_bboxes)  # Get predicted density maps

            # Compute the loss (MSE between predicted and ground truth density maps)
            loss = criterion(pred_dmaps, train_dmaps)

            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.12f}")
            if(epoch == num_epochs-1):
                # Save the final prediction
                save_prediction(train_images, train_dmaps, pred_dmaps, "../predictions/test.png")

if __name__ == "__main__":
    train(num_epochs=50)