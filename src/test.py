import torch
import torch.optim as optim
import torch.nn.functional as F

from network import FamNet
from class_var import DEVICE
from data import Dataset_Creator
from loss import Loss

MODEL_NAME = "Mar_26_11_12_38"

def test_FamNet(learning_rate=10e-7):
    model = FamNet().to(DEVICE)

    # TODO much of this is the same as the trianing loop, could probably pull some out into another function
    # to avoid code duplication

    # Create the dataset and dataloader
    val_data = Dataset_Creator.get_val_dataset()
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)

    pred_counts = []

    for batch_idx, (val_images, val_dmaps, val_bboxes) in enumerate(val_loader):
        # Prepare the data (move to device if using CUDA)
        val_images = val_images.to(DEVICE)
        val_dmaps = val_dmaps.to(DEVICE)
        val_bboxes = val_bboxes.to(DEVICE)

        val_dmaps = val_dmaps.unsqueeze(0) # add extra dimension

        # load the trained weights (reset the training from previous example)
        model.load_state_dict(torch.load(f"../saved_models/{MODEL_NAME}.pth"))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Test time adaptation:
        for step in range(100):

            pred_dmaps = model(val_images, val_bboxes)

            # Ensure pred_dmaps has same shape as train_dmaps
            # Sometimes pred_dmaps is a few pixels off on the width
            if pred_dmaps.shape != val_dmaps.shape:
                pred_dmaps = F.interpolate(pred_dmaps, size=val_dmaps.shape[2:], mode='bilinear', align_corners=False)

            # TODO need to just put the density prediction module params in here
            optimizer.zero_grad()

            # compute adaptation loss
            loss = Loss.adaptation_loss(pred_dmaps, val_bboxes)
            loss.backward()
            optimizer.step()

        pred_dmaps = model(val_images, val_bboxes)
        pred_counts.append(torch.round(torch.sum(pred_dmaps)))

    pred_counts = torch.stack(pred_counts)
    return pred_counts


        




