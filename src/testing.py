import torch

from data import Dataset_Creator
from util import Util
from loss import Loss


# for testing out functionality from other files

if __name__ == "__main__":
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    train_images, train_dmaps, train_bboxes = next(iter(train_loader))

    # test with same dmap
    print(Loss.adaptation_loss(train_dmaps, train_bboxes[0]))