import torch

from network import FamNet
from class_var import DEVICE
from data import Dataset_Creator

MODEL_NAME = "Mar_26_11_12_38"

def test_FamNet():
    model = FamNet().to(DEVICE)
    model.load_state_dict(torch.load(f"../saved_models/{MODEL_NAME}.pth"))

    # Create the dataset and dataloader
    val_data = Dataset_Creator.get_val_dataset()
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True)