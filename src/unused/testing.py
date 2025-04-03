import torch

from data import Dataset_Creator
from util import Util
from loss import Loss

from device import DEVICE


# for testing out functionality from other files

if __name__ == "__main__":
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_bboxes = next(iter(train_loader))
    train_dmaps = train_dmaps.to(DEVICE)

    # test with same dmap
    min_count = Loss._min_count_loss(train_dmaps, train_bboxes[0])
    perturbation = Loss._perturbation_loss(train_dmaps, train_bboxes[0])

    adaptation = Loss.adaptation_loss(train_dmaps, train_bboxes[0])

    print(f"Min_Count: {min_count.item()} | Perturbation: {perturbation.item()} | Adaptation: {adaptation.item()}")