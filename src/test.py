import torch
import torch.optim as optim
import torch.nn.functional as F

from debug import save_prediction
from network import FamNet
from device import DEVICE
from data import Dataset_Creator
from loss import Loss

def test_FamNet(dataset, model_path, learning_rate=10e-7, adaptation=True, limit=None):
    model = FamNet().to(DEVICE)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    total_images = len(dataset) if limit == None else limit

    pred_counts = []

    # load the trained weights
    model.load_state_dict(torch.load(f"../saved_models/{model_path}"))

    for batch_idx, (test_images, test_dmaps, test_bboxes) in enumerate(test_loader):
        # Prepare the data (move to device if using CUDA)
        test_images = test_images.to(DEVICE)
        test_dmaps = test_dmaps.to(DEVICE)
        test_bboxes = test_bboxes.to(DEVICE)

        test_dmaps = test_dmaps.unsqueeze(0) # add extra dimension

        if (adaptation):
            optimizer = optim.Adam(model.density_prediction.parameters(), lr=learning_rate)

            print(f"Adapting Image {batch_idx+1}/{total_images}")

            # Test time adaptation:
            for step in range(100):

                pred_dmaps = model(test_images, test_bboxes)

                # Ensure pred_dmaps has same shape as train_dmaps
                # Sometimes pred_dmaps is a few pixels off on the width
                if pred_dmaps.shape != test_dmaps.shape:
                    pred_dmaps = F.interpolate(pred_dmaps, size=test_dmaps.shape[2:], mode='bilinear', align_corners=False)

                optimizer.zero_grad()

                # compute adaptation loss
                loss = Loss.adaptation_loss(pred_dmaps[0], test_bboxes[0])
                loss.backward()
                optimizer.step()
        
        with torch.no_grad():
            pred_dmaps = model(test_images, test_bboxes)
            # Ensure pred_dmaps has same shape as train_dmaps
            # Sometimes pred_dmaps is a few pixels off on the width
            if pred_dmaps.shape != test_dmaps.shape:
                pred_dmaps = F.interpolate(pred_dmaps, size=test_dmaps.shape[2:], mode='bilinear', align_corners=False)

            pred_count = torch.round(torch.sum(pred_dmaps))
            actual_count = torch.round(torch.sum(test_dmaps))
            criterion = torch.nn.MSELoss()
            loss = criterion(test_dmaps, pred_dmaps)
            pred_counts.append(pred_count)
            print(f"Image {batch_idx+1}/{total_images} | pred_count: {pred_count} | actual_count: {actual_count} | loss: {loss}")
            #save_prediction(test_images, test_dmaps, pred_dmaps, f"../predictions/image{batch_idx}.png")

        if limit != None and batch_idx == limit-1: break

    pred_counts = torch.stack(pred_counts).to(DEVICE)
    return pred_counts
